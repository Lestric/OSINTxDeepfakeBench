# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-29
# description: Data pre-processing script for deepfake dataset.


"""
After running this code, it will generates a json file looks like the below structure for re-arrange data.

{
    "FaceForensics++": {
        "Deepfakes": {
            "video1": {
                "label": "fake",
                "frames": [
                    "/path/to/frames/video1/frame1.png",
                    "/path/to/frames/video1/frame2.png",
                    ...
                ]
            },
            "video2": {
                "label": "fake",
                "frames": [
                    "/path/to/frames/video2/frame1.png",
                    "/path/to/frames/video2/frame2.png",
                    ...
                ]
            },
            ...
        },
        "original_sequences": {
            "youtube": {
                "video1": {
                    "label": "real",
                    "frames": [
                        "/path/to/frames/video1/frame1.png",
                        "/path/to/frames/video1/frame2.png",
                        ...
                    ]
                },
                "video2": {
                    "label": "real",
                    "frames": [
                        "/path/to/frames/video2/frame1.png",
                        "/path/to/frames/video2/frame2.png",
                        ...
                    ]
                },
                ...
            }
        }
    }
}
"""


import os
import glob
import re
import cv2
import json
import yaml
import pandas as pd
from pathlib import Path


def generate_dataset_file(dataset_name, dataset_root_path, output_file_path, compression_level='c23', perturbation = 'end_to_end'):
    """
    Description:
        - Generate a JSON file containing information about the specified datasets' videos and frames.
    Args:
        - dataset: The name of the dataset.
        - dataset_path: The path to the dataset.
        - output_file_path: The path to the output JSON file.
        - compression_level: The compression level of the dataset.
    """

    # Initialize an empty dictionary to store dataset information.
    dataset_dict = {}


    ## FaceForensics++ dataset or DeepfakeDetection dataset
    ## Note: DeepfakeDetection dataset is a subset of FaceForensics++ dataset
    if dataset_name == 'FaceForensics++' or dataset_name.startswith('DeepFakeDetection') or dataset_name == 'FaceShifter': 
        ff_dict = {
            'Deepfakes': 'FF-DF',
            'Face2Face': 'FF-F2F',
            'FaceSwap': 'FF-FS',
            'Real': 'FF-real',
            'DFD_Real': 'DFD_real',
            'NeuralTextures': 'FF-NT',
            'FaceShifter': 'FF-FH',
            'DeepFakeDetection': 'DFD_fake',
            'DeepFakeDetection_original': 'DFD_real',
        }
        # Basispfad an Dataset koppeln
        if dataset_name == 'FaceForensics++':
            dataset_path = os.path.join(dataset_root_path, 'FaceForensics++')
        elif dataset_name.startswith('DeepFakeDetection'):
            dataset_path = os.path.join(dataset_root_path, dataset_name)
        else:  # FaceShifter (als eigener Case unter FF++)
            dataset_path = os.path.join(dataset_root_path, 'FaceForensics++')


        # Nur für FaceForensics++ die Splits laden
        video_to_mode = {}
        if dataset_name == 'FaceForensics++':
            def _load_pairs(p):
                with open(p, 'r') as f:
                    return json.load(f)

            train_json = _load_pairs(os.path.join(dataset_path, 'train.json'))
            val_json   = _load_pairs(os.path.join(dataset_path, 'val.json'))
            test_json  = _load_pairs(os.path.join(dataset_path, 'test.json'))

            # sowohl Einzel-IDs ("071") als auch Kombis ("071_054") abbilden
            for d1, d2 in train_json:
                for a, b in [(d1, None), (d2, None), (d1, d2), (d2, d1)]:
                    key = f"{a}_{b}" if b is not None else a
                    video_to_mode[key] = 'train'
            for d1, d2 in val_json:
                for a, b in [(d1, None), (d2, None), (d1, d2), (d2, d1)]:
                    key = f"{a}_{b}" if b is not None else a
                    video_to_mode[key] = 'val'
            for d1, d2 in test_json:
                for a, b in [(d1, None), (d2, None), (d1, d2), (d2, d1)]:
                    key = f"{a}_{b}" if b is not None else a
                    video_to_mode[key] = 'test'
        else:
            # Für DeepFakeDetection keine Splits laden
            train_json = val_json = test_json = []
            
        # FaceForensics++ real dataset
        if os.path.isdir(dataset_path) and os.path.isdir(os.path.join(dataset_path, 'original_sequences')):
            label = 'Real'
            dataset_dict['FaceForensics++'] = {}
            dataset_dict['FaceForensics++']['FF-real'] = {}
            dataset_dict['FaceForensics++']['DFD_real'] = {'train': {}, 'test': {}, 'val': {}}

            # --- FF++ YouTube-Originals NUR verarbeiten, wenn wir FaceForensics++ bauen UND der Ordner existiert
            youtube_root = os.path.join(dataset_path, 'original_sequences', 'youtube')
            if dataset_name == 'FaceForensics++' and os.path.isdir(youtube_root):
                dataset_dict['FaceForensics++']['FF-real']['train'] = {}
                dataset_dict['FaceForensics++']['FF-real']['test']  = {}
                dataset_dict['FaceForensics++']['FF-real']['val']   = {}
                for comp_dir in os.scandir(youtube_root):
                    if comp_dir.is_dir() and comp_dir.name in ["c23", "c40", "raw"]:
                        compression_level = comp_dir.name
                        dataset_dict['FaceForensics++']['FF-real']['train'][compression_level] = {}
                        dataset_dict['FaceForensics++']['FF-real']['test'][compression_level]  = {}
                        dataset_dict['FaceForensics++']['FF-real']['val'][compression_level]   = {}
                        frames_root = os.path.join(youtube_root, compression_level, 'frames')
                        if os.path.isdir(frames_root):
                            for video_path in os.scandir(frames_root):
                                if video_path.is_dir():
                                    video_name = video_path.name
                                    mode = video_to_mode.get(video_name)
                                    if mode is None:
                                        print(f"[WARN] {video_name} nicht in train/val/test-Listen - überspringe.")
                                        continue
                                    frame_paths = [os.path.join(video_path, f.name) for f in os.scandir(video_path)]
                                    dataset_dict['FaceForensics++']['FF-real'][mode][compression_level][video_name] = {
                                        'label': ff_dict['Real'], 'frames': frame_paths
                                    }

                        
            actors_root = os.path.join(dataset_path, 'original_sequences', 'actors')
            if dataset_name.startswith('DeepFakeDetection') or os.path.isdir(actors_root):
                dataset_dict.setdefault('FaceForensics++', {})
                dataset_dict['FaceForensics++'].setdefault('DFD_real', {'train': {}, 'test': {}, 'val': {}})

                for comp_dir in ('c23', 'c40', 'raw'):
                    frames_root = os.path.join(actors_root, comp_dir, 'frames')
                    if not os.path.isdir(frames_root):
                        continue
                    dataset_dict['FaceForensics++']['DFD_real']['train'].setdefault(comp_dir, {})
                    dataset_dict['FaceForensics++']['DFD_real']['test'].setdefault(comp_dir, {})
                    dataset_dict['FaceForensics++']['DFD_real']['val'].setdefault(comp_dir, {})

                    for video_path in os.scandir(frames_root):
                        if not video_path.is_dir():
                            continue
                        video_name = video_path.name
                        frame_paths = [os.path.join(video_path, f.name) for f in os.scandir(video_path)]
                        # Für DFD_real keine offiziellen Splits ? alles in train/test/val spiegeln
                        dataset_dict['FaceForensics++']['DFD_real']['train'][comp_dir][video_name] = {'label': ff_dict['DFD_Real'], 'frames': frame_paths}
                        dataset_dict['FaceForensics++']['DFD_real']['test'][comp_dir][video_name]  = {'label': ff_dict['DFD_Real'], 'frames': frame_paths}
                        dataset_dict['FaceForensics++']['DFD_real']['val'][comp_dir][video_name]   = {'label': ff_dict['DFD_Real'], 'frames': frame_paths}
            else:
                # wenn actors nicht existiert und nicht DeepFakeDetection: sauber überspringen
                pass
        # FaceForensics++ fake datasets
        if os.path.isdir(os.path.join(dataset_path, 'manipulated_sequences')):
            dataset_dict.setdefault('FaceForensics++', {})
            for label_dir in os.scandir(os.path.join(dataset_path, 'manipulated_sequences')):
                if label_dir.is_dir():
                    label = label_dir.name
                    dataset_dict['FaceForensics++'][ff_dict[label]] = {}
                    dataset_dict['FaceForensics++'][ff_dict[label]]['train'] = {}
                    dataset_dict['FaceForensics++'][ff_dict[label]]['test'] = {}
                    dataset_dict['FaceForensics++'][ff_dict[label]]['val'] = {}
                    
                    # Iterate over all compression levels: c23, c40, raw
                    for compression_level in os.scandir(os.path.join(dataset_path, 'manipulated_sequences', label)):
                        if compression_level.is_dir() and compression_level.name in ["c23", "c40", "raw"]:
                            compression_level = compression_level.name
                            dataset_dict['FaceForensics++'][ff_dict[label]]['train'][compression_level] = {}
                            dataset_dict['FaceForensics++'][ff_dict[label]]['test'][compression_level] = {}
                            dataset_dict['FaceForensics++'][ff_dict[label]]['val'][compression_level] = {}
                            # Iterate over all videos

                            for video_path in os.scandir(os.path.join(dataset_path, 'manipulated_sequences', label, compression_level, 'frames')):
                                if video_path.is_dir():
                                    video_name = video_path.name
                                    frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                                    if label != 'FaceShifter':
                                        mask_paths = os.path.join(dataset_path, 'manipulated_sequences', label, 'c23','masks', video_name)
                                        # mask is all the same for all compression levels
                                        if os.path.exists(mask_paths):
                                            mask_frames_paths = [os.path.join(mask_paths, frame.name) for frame in os.scandir(mask_paths)]
                                        else:
                                            mask_frames_paths = []
                                        try:
                                            mode = video_to_mode[video_name]
                                            dataset_dict['FaceForensics++'][ff_dict[label]][mode][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths, 'masks': mask_frames_paths}
                                        # DeepfakeDetection dataset
                                        except:
                                            dataset_dict['FaceForensics++'][ff_dict[label]]['train'][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths, 'masks': mask_frames_paths}
                                            dataset_dict['FaceForensics++'][ff_dict[label]]['val'][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths, 'masks': mask_frames_paths}
                                            dataset_dict['FaceForensics++'][ff_dict[label]]['test'][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths, 'masks': mask_frames_paths}
                                    # FaceShifter dataset
                                    else:
                                        mode = video_to_mode[video_name]
                                        dataset_dict['FaceForensics++'][ff_dict[label]][mode][compression_level][video_name] = {'label': ff_dict[label], 'frames': frame_paths}
         

        # get the DeepfakeDetection dataset from FaceForensics++ dataset
        if dataset_name == 'FaceForensics++':
            # Delete the DeepfakeDetection dataset from FaceForensics++ dataset
            ffpp_root = dataset_dict.get('FaceForensics++', {})
            for k in ('DFD_fake', 'DFD_real', 'FF-FH'):
                ffpp_root.pop(k, None)
        elif dataset_name.startswith('DeepFakeDetection'):
            # Check if the DeepfakeDetection dataset is in the FaceForensics++ dataset
            if 'DFD_fake' in dataset_dict['FaceForensics++'] and \
                'DFD_real' in dataset_dict['FaceForensics++']:
                # Add the DeepfakeDetection dataset to the dataset_dict
                dataset_dict[dataset_name] = {
                    'DFD_fake': dataset_dict['FaceForensics++']['DFD_fake'], 
                    'DFD_real': dataset_dict['FaceForensics++']['DFD_real']
                }
                del dataset_dict['FaceForensics++']
        elif dataset_name == 'FaceShifter':
            if 'FF-FH' in dataset_dict['FaceForensics++'] and \
                'FF-real' in dataset_dict['FaceForensics++']:
                # Add the DeepfakeDetection dataset to the dataset_dict
                dataset_dict['FaceShifter'] = {
                    'FF-FH': dataset_dict['FaceForensics++']['FF-FH'], 
                    'FF-real': dataset_dict['FaceForensics++']['FF-real']
                }
                del dataset_dict['FaceForensics++']
            else:
                # TODO
                raise ValueError('DeepfakeDetection dataset not found in FaceForensics++ dataset.')
        else:
            raise ValueError('Invalid dataset name: {}'.format(dataset_name))

        # if FaceForensics++, based on label and generate the json
        if dataset_name == 'FaceForensics++':
            for label, value in dataset_dict['FaceForensics++'].items():
                if label != 'FF-real':
                    with open(os.path.join(output_file_path,f'{label}.json'), 'w') as f:
                        data = {label: {'FF-real': dataset_dict['FaceForensics++']['FF-real'],
                                        label: value,
                                        }}
                        json.dump(data, f)
                        print(f"Finish writing {label}.json")
    
    ## Celeb-DF-v1 dataset
    ## Note: videos in Celeb-DF-v1/2 are not in the same format as in FaceForensics++ dataset
    elif dataset_name == 'Celeb-DF-v1':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            if folder.name in ['Celeb-real', 'YouTube-real']:
                label = 'CelebDFv1_real'
            else:
                label = 'CelebDFv1_fake'
            assert label in ['CelebDFv1_real', 'CelebDFv1_fake'], 'Invalid label: {}'.format(label)
            dataset_dict[dataset_name][label] = {}
            dataset_dict[dataset_name][label]['train'] = {}
            dataset_dict[dataset_name][label]['val'] = {}
            dataset_dict[dataset_name][label]['test'] = {}
            for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                if video_path.is_dir():
                    video_name = video_path.name
                    frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                    dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
        
        # Special case for test&val data of Celeb-DF-v1/2
        with open(os.path.join(dataset_root_path, dataset_name, 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'CelebDFv1_real'
            elif 'synthesis' in line:
                label = 'CelebDFv1_fake'
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
            
            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join(dataset_root_path, dataset_name, line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}

    ## Celeb-DF-v2 dataset
    ## Note: videos in Celeb-DF-v1/2 are not in the same format as in FaceForensics++ dataset
    elif dataset_name.startswith('Celeb-DF-v2'):
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            if folder.name in ['Celeb-real', 'YouTube-real']:
                label = 'CelebDFv2_real'
            else:
                label = 'CelebDFv2_fake'
            assert label in ['CelebDFv2_real', 'CelebDFv2_fake'], 'Invalid label: {}'.format(label)
            dataset_dict[dataset_name][label] = {}
            dataset_dict[dataset_name][label]['train'] = {}
            dataset_dict[dataset_name][label]['val'] = {}
            dataset_dict[dataset_name][label]['test'] = {}
            for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                if video_path.is_dir():
                    video_name = video_path.name
                    frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                    dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
        
        # Special case for test&val data of Celeb-DF-v1/2
        with open(os.path.join(dataset_root_path, dataset_name, 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'CelebDFv2_real'
            elif 'synthesis' in line:
                label = 'CelebDFv2_fake'
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
            
            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join(dataset_root_path, dataset_name, line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}

    ## DFDCP dataset
    elif dataset_name == 'DFDCP':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        #initialize the dataset dictionary
        dataset_dict[dataset_name] = {'DFDCP_Real': {'train': {}, 'test': {}, 'val': {}},
                                'DFDCP_FakeA': {'train': {}, 'test': {}, 'val': {}},
                                'DFDCP_FakeB': {'train': {}, 'test': {}, 'val': {}}}
        # Open the dataset information file ('dataset.json') and parse its contents
        with open(os.path.join(dataset_path, 'dataset.json' ), 'r') as f:
            dataset_info = json.load(f)
        # Iterate over the dataset_info dictionary and extract the index and file name for each video
        for dataset in dataset_info.keys():
            index = dataset.split('/')[0]
            vidname = dataset.split('/')[-1].split(".")[0]
            if Path(os.path.join(dataset_path, index, 'frames', vidname)).exists():
                frame_paths = glob.glob(os.path.join(dataset_path, index, 'frames', vidname, '*png'))
                if len(frame_paths) == 0:
                    continue
                label = dataset_info[dataset]['label']
                if label == 'real':
                    label = 'DFDCP_Real'
                elif label == 'fake' and index == 'method_A':
                    label = 'DFDCP_FakeA'
                elif label == 'fake' and index == 'method_B':
                    label = 'DFDCP_FakeB'
                else:
                    raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
                set_attr = dataset_info[dataset]['set']  # train, test, val
                dataset_dict[dataset_name][label][set_attr][vidname] = {'label': label, 'frames': frame_paths}
        # Special case for val data of DFDCP
        for label in ['DFDCP_Real', 'DFDCP_FakeA', 'DFDCP_FakeB']:
            dataset_dict[dataset_name][label]['val'] = dataset_dict[dataset_name][label]['test']
    
    ## DFDC dataset
    elif dataset_name == 'DFDC':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {'DFDC_Real': {'train': {}, 'test': {}, 'val': {}},
                                'DFDC_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            if folder.name in ['test']:
                # 读取csv文件
                df = pd.read_csv(os.path.join(dataset_path,folder.name,'labels.csv'))
                labels = ['DFDC_Real','DFDC_Fake']
                # 循环遍历每一行，并逐行读取filename和label的值
                for index, row in df.iterrows():
                    vidname = row['filename'].split('.mp4')[0]
                    label = labels[row['label']]
                    assert label in ['DFDC_Real','DFDC_Fake'], 'Invalid label: {}'.format(label)
                    frame_paths = glob.glob(os.path.join(dataset_path, folder.name,'frames', vidname, '*png'))
                    if len(frame_paths) == 0:
                        continue
                    dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
                    dataset_dict[dataset_name][label]['val'] = {'label': label, 'frames': frame_paths}
            
            elif folder.name in ['train']:
                num_file = 0
                for dfdc_train_part in os.scandir(os.path.join(dataset_path, folder.name)):
                    if not os.path.isdir(dfdc_train_part):
                        continue
                    num_file += 1
                    print('processing {}th file in 50 files.'.format(num_file))
                    with open(os.path.join(dfdc_train_part, 'metadata.json'), 'r') as f:
                            metadata = json.load(f)
                    for video_path in os.scandir(os.path.join(dfdc_train_part, 'frames')):
                        if video_path.is_dir():
                            video_name = video_path.name
                            label = metadata[video_name + ".mp4"]["label"]
                            assert label in ['REAL', 'FAKE'], 'Invalid label: {}'.format(label)
                            if label == 'REAL':
                                label = 'DFDC_Real'
                            else:
                                label = 'DFDC_Fake'
                            frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                            dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                            dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

    ## DeeperForensics-1.0 dataset
    elif dataset_name == 'DeeperForensics-1.0':
        with open(os.path.join(dataset_root_path, dataset_name, 'lists/splits/train.txt'), 'r') as f:
            train_txt = f.readlines()
            train_txt = [line.strip().split('.')[0] for line in train_txt]
        with open(os.path.join(dataset_root_path, dataset_name, 'lists/splits/test.txt'), 'r') as f:
            test_txt = f.readlines()
            test_txt = [line.strip().split('.')[0] for line in test_txt]
        with open(os.path.join(dataset_root_path, dataset_name, 'lists/splits/val.txt'), 'r') as f:
            val_txt = f.readlines()
            val_txt = [line.strip().split('.')[0] for line in val_txt]
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {'DF_real': {'train': {}, 'test': {}, 'val': {}},
                                'DF_fake': {'train': {}, 'test': {}, 'val': {}}}
        if not Path(os.path.join(dataset_path, 'manipulated_videos', perturbation)).exists():
            raise ValueError(f"wrong in processing perturbation {perturbation} in manipulated_videos")
        print(f"processing perturbation {perturbation} in manipulated_videos")
        for video_path in os.scandir(os.path.join(dataset_path, 'manipulated_videos', perturbation, 'frames')):
            if video_path.is_dir():
                video_name = video_path.name
                if video_name in train_txt:
                    set_attr = 'train'
                elif video_name in test_txt:
                    set_attr = 'test'
                elif video_name in val_txt:
                    set_attr = 'val'
                else:
                    raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")
                label = 'DF_fake'
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                ## if frame image in frame_paths is not the correct png, skip this frame yxh
                for frame_path in frame_paths:
                    if cv2.imread(frame_path) is None:
                        frame_paths.remove(frame_path)
                dataset_dict[dataset_name][label][set_attr][video_name] = {'label': label, 'frames': frame_paths}
        for actor_path in os.scandir(os.path.join(dataset_path, 'source_videos')):
            print("actor",actor_path.name)
            if not os.path.isdir(actor_path):
                continue
            label = 'DF_real'
            video_paths = [os.path.join(actor_path, 'frames', video.name) for video in os.scandir(os.path.join(actor_path, 'frames'))]
            for video_path in video_paths:
                video_name = video_path.split('/')[-1]
                frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                ## if frame image in frame_paths is not the correct png, skip this frame yxh
                for frame_path in frame_paths:
                    if cv2.imread(frame_path) is None:
                        frame_paths.remove(frame_path)
                dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}
        
    ## UADFV dataset
    elif dataset_name == 'UADFV':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {'UADFV_Real': {'train': {}, 'test': {}, 'val': {}},
                                'UADFV_Fake': {'train': {}, 'test': {}, 'val': {}}}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            elif folder.name in ['fake']:
                for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                    if video_path.is_dir():
                        video_name = video_path.name
                        label = 'UADFV_Fake'
                        frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                        dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}
            elif folder.name in ['real']:
                for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                    if video_path.is_dir():
                        video_name = video_path.name
                        label = 'UADFV_Real'
                        frame_paths = [os.path.join(video_path, frame.name) for frame in os.scandir(video_path)]
                        dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

    # Convert the dataset dictionary to JSON format and save to file
    output_file_path = os.path.join(output_file_path, dataset_name + '.json')
    with open(output_file_path, 'w') as f:
        json.dump(dataset_dict, f)
    # print the successfully generated dataset dictionary
    print(f"{dataset_name}.json generated successfully.")

if __name__ == '__main__':
    # from config.yaml load parameters
    yaml_path = './config.yaml'
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    dataset_name = config['rearrange']['dataset_name']['default']
    dataset_root_path = config['rearrange']['dataset_root_path']['default']
    output_file_path = config['rearrange']['output_file_path']['default']
    comp = config['rearrange']['comp']['default']
    perturbation = config['rearrange']['perturbation']['default']
    # Call the generate_dataset_file function
    generate_dataset_file(dataset_name, dataset_root_path, output_file_path, comp, perturbation)
