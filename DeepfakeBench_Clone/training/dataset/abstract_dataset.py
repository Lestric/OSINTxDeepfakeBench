# author: Zhiyuan Yan (modifiziert)
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.
# notes (mod): LMDB robust fixes: lazy-open per worker, readahead=False, retry on MemoryError

import sys
sys.path.append('.')

import os
import math
import yaml
import glob
import json
import random
from copy import deepcopy
from collections import defaultdict

import lmdb
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T

import albumentations as A
from .albu import IsotropicResize


FFpp_pool = ['FaceForensics++','FaceShifter','FF-DF','FF-F2F','FF-FS','FF-NT']


def all_in_pool(inputs, pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        """
        Args:
            config (dict): configuration parameters.
            mode (str): 'train' or 'test'
        """
        if mode not in ('train', 'test'):
            raise NotImplementedError('Only train and test modes are supported.')

        self.config = config
        self.mode = mode

        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb_enabled = config.get('lmdb', False)

        # optional debug flag
        self.debug_lmdb = bool(config.get('debug_lmdb', False))

        # will be set below if lmdb_enabled
        self.lmdb_path = None
        self.env = None  # IMPORTANT: lazy-open per worker

        # Collect file lists
        image_list, label_list = [], []
        if mode == 'train':
            dataset_list = config['train_dataset']
            for one_data in dataset_list:
                tmp_image, tmp_label, _ = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
            if self.lmdb_enabled:
                if len(dataset_list) > 1:
                    if all_in_pool(dataset_list, FFpp_pool):
                        self.lmdb_path = os.path.join(config['lmdb_dir'], "FaceForensics++_lmdb")
                    else:
                        raise ValueError('Training with multiple dataset and LMDB is not implemented yet.')
                else:
                    ds = dataset_list[0]
                    self.lmdb_path = os.path.join(config['lmdb_dir'], f"{ds if ds not in FFpp_pool else 'FaceForensics++'}_lmdb")
        else:  # test
            one_data = config['test_dataset']
            image_list, label_list, _ = self.collect_img_and_label_for_one_dataset(one_data)
            if self.lmdb_enabled:
                self.lmdb_path = os.path.join(
                    config['lmdb_dir'],
                    f"{one_data}_lmdb" if one_data not in FFpp_pool else 'FaceForensics++_lmdb'
                )

        assert len(image_list) and len(label_list), f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list
        self.data_dict = {'image': self.image_list, 'label': self.label_list}

        # Albumentations pipeline
        self.transform = self.init_data_aug_method()

        # just an info print (env wird NICHT hier geöffnet)
        if self.lmdb_enabled and self.debug_lmdb:
            print(f"[LMDB] path set to: {self.lmdb_path} (lazy-open per worker)")

    # ---------------- LMDB helpers ----------------

    def _open_env_if_needed(self):
        """Open LMDB environment lazily (per worker)."""
        if (not self.lmdb_enabled) or (self.env is not None):
            return
        # safer defaults for dataloader workers
        self.env = lmdb.open(
            self.lmdb_path,
            create=False,
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,       # reduce RSS for random reads
            max_readers=4096
        )
        if self.debug_lmdb:
            print(f"[LMDB] opened in PID={os.getpid()} path={self.lmdb_path}")

    def _close_env(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            finally:
                self.env = None

    def _reopen_env_on_error(self):
        """Close and reopen env once (used on MemoryError)."""
        self._close_env()
        self._open_env_if_needed()

    # ---------------- Albumentations ----------------

    def init_data_aug_method(self):
        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'],
                     p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'],
                           p=self.config['data_aug']['blur_prob']),
            A.OneOf([
                IsotropicResize(max_side=self.config['resolution'],
                                interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'],
                                interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'],
                                interpolation_down=cv2.INTER_LINEAR,
                                interpolation_up=cv2.INTER_LINEAR),
            ], p=0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['data_aug']['brightness_limit'],
                    contrast_limit=self.config['data_aug']['contrast_limit']
                ),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(
                quality_lower=self.config['data_aug']['quality_lower'],
                quality_upper=self.config['data_aug']['quality_upper'],
                p=0.5
            )
        ],
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    # ---------------- Collect file lists ----------------

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        return rescaled_landmarks

    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        label_list = []
        frame_path_list = []
        video_name_list = []

        # dataset json
        if not os.path.exists(self.config['dataset_json_folder']):
            self.config['dataset_json_folder'] = self.config['dataset_json_folder'].replace(
                '/Youtu_Pangu_Security_Public', '/Youtu_Pangu_Security/public'
            )
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        # ugly but consistent with original
        cp = None
        if dataset_name == 'FaceForensics++_c40':
            dataset_name, cp = 'FaceForensics++', 'c40'
        elif dataset_name == 'FF-DF_c40':
            dataset_name, cp = 'FF-DF', 'c40'
        elif dataset_name == 'FF-F2F_c40':
            dataset_name, cp = 'FF-F2F', 'c40'
        elif dataset_name == 'FF-FS_c40':
            dataset_name, cp = 'FF-FS', 'c40'
        elif dataset_name == 'FF-NT_c40':
            dataset_name, cp = 'FF-NT', 'c40'

        for label in dataset_info[dataset_name]:
            # split handling (allows overriding to 'train' etc.)
            split = self.config.get('eval_split', self.mode)
            sub_dataset_info = dataset_info[dataset_name][label][split]

            # compression choice
            if cp is None and dataset_name in [
                'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                'DeepFakeDetection', 'DeepFakeDetection-FACE',
                'DeepFakeDetection-S_W', 'DeepFakeDetection-TEXT',
                'DeepFakeDetection-JPEG', 'FaceShifter'
            ]:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == 'c40' and dataset_name in [
                'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                'DeepFakeDetection', 'FaceShifter'
            ]:
                sub_dataset_info = sub_dataset_info['c40']

            for video_name, video_info in sub_dataset_info.items():
                unique_video_name = video_info['label'] + '_' + video_name

                if video_info['label'] not in self.config['label_dict']:
                    raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                lb = self.config['label_dict'][video_info['label']]
                frame_paths = video_info['frames']

                if '\\' in frame_paths[0]:
                    frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                else:
                    frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))

                total_frames = len(frame_paths)
                # reduce to frame_num if necessary
                if self.frame_num < total_frames:
                    total_frames = self.frame_num
                    if self.video_level:
                        start_frame = random.randint(0, total_frames - self.frame_num) if self.mode == 'train' else 0
                        frame_paths = frame_paths[start_frame:start_frame + self.frame_num]
                    else:
                        step = total_frames // self.frame_num
                        frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]

                # video-level: clips
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError('clip_size must be specified when video_level is True.')

                    if total_frames >= self.clip_size:
                        selected_clips = []
                        num_clips = total_frames // self.clip_size
                        if num_clips > 1:
                            clip_step = (total_frames - self.clip_size) // (num_clips - 1)
                            for i in range(num_clips):
                                start_frame = (random.randrange(i * clip_step, min((i + 1) * clip_step, total_frames - self.clip_size + 1))
                                               if self.mode == 'train' else i * clip_step)
                                continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                assert len(continuous_frames) == self.clip_size
                                selected_clips.append(continuous_frames)
                        else:
                            start_frame = random.randrange(0, total_frames - self.clip_size + 1) if self.mode == 'train' else 0
                            continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                            assert len(continuous_frames) == self.clip_size
                            selected_clips.append(continuous_frames)

                        label_list.extend([lb] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        video_name_list.extend([unique_video_name] * len(selected_clips))
                    else:
                        print(f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames}).")
                else:
                    # image-level
                    label_list.extend([lb] * total_frames)
                    frame_path_list.extend(frame_paths)
                    video_name_list.extend([unique_video_name] * len(frame_paths))

        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)
        return frame_path_list, label_list, video_name_list

    # ---------------- Path -> LMDB key ----------------

    def _lmdb_key_from_abs_rgb(self, abs_path: str) -> str:
        """
        Map absolute path like
          /home/user/datasets/rgb/FaceForensics++/...
        to LMDB key:
          FaceForensics++/...
        """
        p = os.path.abspath(abs_path).replace('\\', '/')
        anchor = '/rgb/'
        i = p.lower().find(anchor)
        if i != -1:
            key = p[i + len(anchor):]
        else:
            rgb_root = str(self.config.get('rgb_dir', '')).replace('\\', '/').rstrip('/') + '/'
            if rgb_root != '/' and p.startswith(rgb_root):
                key = p[len(rgb_root):]
            else:
                key = p
        return key

    # ---------------- I/O: RGB / MASK / LANDMARK ----------------

    def load_rgb(self, file_path):
        """
        Returns PIL.Image (RGB) resized to config['resolution'].
        """
        size = self.config['resolution']
        try:
            if not self.lmdb_enabled:
                if self.debug_lmdb:
                    print(f"[RGB-FS] path={file_path}")
                assert os.path.exists(file_path), f"File not found: {file_path}"
                img = cv2.imread(file_path)
                if img is None:
                    raise FileNotFoundError(f"cv2.imread returned None for: {file_path}")
            else:
                self._open_env_if_needed()
                key = self._lmdb_key_from_abs_rgb(file_path)
                if self.debug_lmdb:
                    print(f"[RGB-LMDB] key={key}")
                # attempt 1
                try:
                    with self.env.begin(write=False) as txn:
                        image_bin = txn.get(key.encode('utf-8'))
                except MemoryError:
                    # reopen once and retry
                    self._reopen_env_on_error()
                    with self.env.begin(write=False) as txn:
                        image_bin = txn.get(key.encode('utf-8'))

                if image_bin is None:
                    raise KeyError(f"LMDB key not found: {key}")

                buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"cv2.imdecode failed for LMDB key: {key}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
            return Image.fromarray(np.array(img, dtype=np.uint8))
        except Exception as e:
            # add context
            raise RuntimeError(f"load_rgb failed (lmdb={self.lmdb_enabled}) for path={file_path}") from e

    def load_mask(self, file_path):
        """
        Returns float32 mask HxWx1 in [0,1] (or zeros), resized to config['resolution'].
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1), dtype=np.float32)

        if not self.lmdb_enabled:
            if not file_path[0] == '.':
                file_path = f'./{self.config["rgb_dir"]}\\' + file_path
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size), dtype=np.uint8)
            else:
                return np.zeros((size, size, 1), dtype=np.float32)
        else:
            self._open_env_if_needed()
            try:
                if file_path[0] == '.':
                    file_path = file_path.replace('./datasets\\', '')
                with self.env.begin(write=False) as txn:
                    image_bin = txn.get(file_path.encode())
            except MemoryError:
                self._reopen_env_on_error()
                with self.env.begin(write=False) as txn:
                    image_bin = txn.get(file_path.encode())

            if image_bin is None:
                mask = np.zeros((size, size, 3), dtype=np.uint8)
            else:
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)  # keep 3ch then collapse below

        mask = cv2.resize(mask, (size, size)) / 255.0
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Returns (81,2) float32 array (scaled), or zeros.
        """
        if file_path is None:
            return np.zeros((81, 2), dtype=np.float32)

        if not self.lmdb_enabled:
            if not file_path[0] == '.':
                file_path = f'./{self.config["rgb_dir"]}\\' + file_path
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2), dtype=np.float32)
        else:
            self._open_env_if_needed()
            try:
                if file_path[0] == '.':
                    file_path = file_path.replace('./datasets\\', '')
                with self.env.begin(write=False) as txn:
                    binary = txn.get(file_path.encode())
            except MemoryError:
                self._reopen_env_on_error()
                with self.env.begin(write=False) as txn:
                    binary = txn.get(file_path.encode())

            if binary is None:
                return np.zeros((81, 2), dtype=np.float32)
            landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
            landmark = self.rescale_landmarks(np.float32(landmark), original_size=256,
                                              new_size=self.config['resolution'])
        return landmark

    # ---------------- Tensor utils ----------------

    def to_tensor(self, img):
        return T.ToTensor()(img)

    def normalize(self, img):
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        kwargs = {'image': img}
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask

        transformed = self.transform(**kwargs)
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask', mask)

        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    # ---------------- PyTorch dataset API ----------------

    def __getitem__(self, index, no_norm=False):
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            mask_path = image_path.replace('frames', 'masks')
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')

            # robust load, raise with full stack if fails
            try:
                image = self.load_rgb(image_path)
            except Exception:
                import traceback
                print(f"[GETITEM] index={index} path={image_path}")
                traceback.print_exc()
                raise

            image = np.array(image)  # for albumentations

            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark'] and landmarks is not None:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask'] and mask_trans is not None:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            image_tensors = torch.stack(image_tensors, dim=0)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            image_tensors = image_tensors[0]
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors

    @staticmethod
    def collate_fn(batch):
        images, labels, landmarks, masks = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        data_dict = {
            'image': images,
            'label': labels,
            'landmark': landmarks,
            'mask': masks
        }
        return data_dict

    def __len__(self):
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)

    def __del__(self):
        # ensure LMDB is closed when dataset object is GC'ed
        self._close_env()


if __name__ == "__main__":
    # Beispiel: nicht produktiv genutzt; hier keine LMDB-Öffnung
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/video_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(config=config, mode='train')
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config['train_batchSize'],
        shuffle=True,
        num_workers=0,
        collate_fn=train_set.collate_fn,
    )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        ...

