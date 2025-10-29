"""
vortrainiertes Modell immer Effort in dem Fall (erweitert um per-Bild Vorhersagen, ohne Score-Inversion).
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger
import json
import csv

# === Feste Werte (ehemalige CLI-Argumente) ===
EXP = 'gen'          # statt --exp
TAG = 'baseline'     # statt --tag

# === Argument-Parser (ohne --exp und --tag) ===
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='./training/config/detector/xception.yaml',
                    help='Pfad zur Detector-YAML')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='./weights/effort_clip_L14_trainOn_FaceForensic_fixed.pth',
                    help='Pfad zu den Modellgewichten (.pth)')

# Ausgabesteuerung
parser.add_argument('--metrics_outdir', type=str, default='analysis_outputs/metrics',
                    help='Ablageort für JSON/NPY/CSV-Metriken')

# per-Bild-Ausgaben
parser.add_argument('--dump_csv', action='store_true',
                    help='CSV mit per-Bild-Vorhersagen schreiben')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Schwelle für Positivklasse nach Wahrscheinlichkeitsbildung')
parser.add_argument('--preview', type=int, default=10,
                    help='Wie viele Vorhersagen in der Konsole zeigen')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        config = config.copy()
        config['test_dataset'] = test_name
        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='test',
        )
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set, 
            batch_size=config['test_batchSize'],
            shuffle=False, 
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=False
        )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


@torch.no_grad()
def inference(model, data_dict):
    return model(data_dict, inference=True)


# ====== robuste Konversion zu Wahrscheinlichkeiten ======
def _to_prob(pred):
    import numpy as _np
    import torch as _torch

    if isinstance(pred, dict):
        if 'prob' in pred and pred['prob'] is not None:
            t = pred['prob']
            if isinstance(t, _torch.Tensor):
                t = t.detach().cpu().numpy()
            else:
                t = _np.asarray(t)
            return t if t.ndim == 1 else t[:, 1]

        t = None
        for k in ('logits', 'feat', 'emb'):
            v = pred.get(k, None)
            if v is not None:
                t = v
                break

        if t is None:
            raise ValueError("Predictions-Dict enthält weder 'prob' noch 'logits/feat/emb'.")

        if isinstance(t, _torch.Tensor):
            t = t.detach().cpu().numpy()
        else:
            t = _np.asarray(t)

        if t.ndim == 1:
            return 1.0 / (1.0 + _np.exp(-t))
        elif t.ndim == 2 and t.shape[1] == 2:
            e = _np.exp(t - t.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            return p[:, 1].astype(_np.float32)
        else:
            flat = t.reshape(t.shape[0], -1).mean(axis=1)
            return (1.0 / (1.0 + _np.exp(-flat))).astype(_np.float32)

    t = pred
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    else:
        t = np.asarray(t)

    if t.ndim == 1:
        if np.nanmin(t) < 0.0 or np.nanmax(t) > 1.0:
            return 1.0 / (1.0 + np.exp(-t))
        return t.astype(np.float32)
    elif t.ndim == 2 and t.shape[1] == 2:
        if np.any(t < 0) or np.any(t > 1):
            e = np.exp(t - t.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            return p[:, 1].astype(np.float32)
        return t[:, 1].astype(np.float32)

    flat = t.reshape(t.shape[0], -1).mean(axis=1)
    return (1.0 / (1.0 + _np.exp(-flat))).astype(_np.float32)


def _final_score_to_pred(score, threshold=0.5):
    s = score
    return (s >= threshold).astype(np.int64), s


def _pred_paths(outdir, detector_name, dataset_name):
    base = f"{detector_name}__{dataset_name}__{EXP}-{TAG}"
    y_true_path  = os.path.join(outdir, base + "_y_true.npy")
    y_score_path = os.path.join(outdir, base + "_y_score.npy")
    feat_path    = os.path.join(outdir, base + "_feat.npy")
    return y_true_path, y_score_path, feat_path


def _dump_metrics_json(detector_name, dataset_name, metrics, used, total, outdir, mode="frame", y_true_path=None, y_score_path=None, feat_path=None):
    os.makedirs(outdir, exist_ok=True)
    payload = {
        "detector": detector_name,
        "dataset": dataset_name,
        "exp": EXP,
        "tag": TAG,
        "mode": mode,
        "count_total": int(total),
        "count_used": int(used),
        "metrics": {
            "auc": float(metrics.get('auc', 0.0)),
            "acc": float(metrics.get('acc', 0.0)),
            "eer": float(metrics.get('eer', 0.0)),
            "ap":  float(metrics.get('ap',  0.0)),
        },
        "y_true_path": y_true_path,
        "y_score_path": y_score_path,
        "feat_path": feat_path,
    }
    out_path = os.path.join(outdir, f"{detector_name}__{dataset_name}__{EXP}-{TAG}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _sample_paths_from_dataset(dl, batch_index, batch_size):
    start = batch_index * batch_size
    end = start + batch_size
    arr = dl.dataset.image_list[start:end]

    def _norm_path(x):
        if isinstance(x, list) and len(x):
            return x[0]
        return x

    paths = [ _norm_path(e) for e in arr ]
    paths = [ p if isinstance(p, str) else "<unknown_path>" for p in paths ]
    return paths


def test_one_dataset(model, data_loader):
    prediction_scores = []
    feature_lists = []
    label_lists = []
    image_paths = []

    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        data, label, mask, landmark = \
            data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']

        label = torch.where(data_dict['label'] != 0, 1, 0)
        batch_paths = _sample_paths_from_dataset(data_loader, i, data.size(0))
        image_paths.extend(batch_paths)

        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        pred_out = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().numpy())
        score = _to_prob(pred_out)
        prediction_scores += list(score)

        if isinstance(pred_out, dict):
            f = None
            for k in ('emb', 'feat', 'logits'):
                v = pred_out.get(k, None)
                if v is not None:
                    f = v
                    break
            if f is not None:
                if isinstance(f, torch.Tensor):
                    f = f.detach().cpu()
                if f.ndim == 4:
                    f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
                elif f.ndim > 2:
                    f = f.flatten(1)
                feature_lists += list(f.numpy())

    y_score_np = np.array(prediction_scores, dtype=np.float32)
    y_true_np  = np.array(label_lists,       dtype=np.int64)
    feat_np    = np.array(feature_lists) if len(feature_lists) else None
    return y_score_np, y_true_np, feat_np, image_paths


def test_epoch(model, test_data_loaders):
    model.eval()
    metrics_all_datasets = {}
    detector_name = getattr(model, 'config', {}).get('model_name', 'detector')

    for key in test_data_loaders.keys():
        dl = test_data_loaders[key]
        data_dict_ds = dl.dataset.data_dict

        predictions_nps, label_nps, feat_nps, img_names = test_one_dataset(model, dl)
        y_pred_bin, y_score_final = _final_score_to_pred(predictions_nps, threshold=args.threshold)

        n_prev = min(args.preview, len(y_score_final))
        print("[Preview] Erste {} Beispiele:".format(n_prev))
        for i in range(n_prev):
            path_i = img_names[i] if i < len(img_names) else "<unknown>"
            dec = "FAKE" if int(y_pred_bin[i]) == 1 else "REAL"
            print(f"  {i:03d}  path={path_i}  score={y_score_final[i]:.3f}  pred={int(y_pred_bin[i])} ({dec})  true={int(label_nps[i])}")

        if args.dump_csv:
            os.makedirs(args.metrics_outdir, exist_ok=True)
            csv_path = os.path.join(
                args.metrics_outdir,
                f"predictions__{detector_name}__{key}__{EXP}-{TAG}.csv"
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
                w = csv.writer(fcsv)
                w.writerow(["image_path", "y_true", "y_score", "y_pred", "decision"])
                for p, yt, ys, yp in zip(img_names, label_nps, y_score_final, y_pred_bin):
                    dec = "FAKE" if yp == 1 else "REAL"
                    w.writerow([p, int(yt), f"{ys:.6f}", int(yp), dec])
            print(f"[CSV] Per-Bild-Vorhersagen gespeichert: {csv_path}")

        os.makedirs(args.metrics_outdir, exist_ok=True)
        y_true_path, y_score_path, feat_path = _pred_paths(args.metrics_outdir, detector_name, key)

        np.save(y_true_path, np.asarray(label_nps, dtype=np.int64).reshape(-1))
        np.save(y_score_path, np.asarray(y_score_final, dtype=np.float32))

        saved_feat_path = None
        if feat_nps is not None and len(feat_nps):
            feats = np.asarray(feat_nps)
            np.save(feat_path, feats)
            saved_feat_path = feat_path

        metric_one_dataset = get_test_metrics(y_pred=y_score_final, y_true=label_nps,
                                              img_names=data_dict_ds['image'])
        metrics_all_datasets[key] = metric_one_dataset

        num_used = int(len(label_nps))
        num_total = int(len(dl.dataset.image_list))
        print(f"[{key}] Bilder im Test: verwendet={num_used} / gesamt={num_total}")

        _dump_metrics_json(
            detector_name, key, metric_one_dataset, num_used, num_total,
            args.metrics_outdir, mode="frame",
            y_true_path=y_true_path, y_score_path=y_score_path, feat_path=saved_feat_path
        )

    return metrics_all_datasets


def main():
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']

    weights_path = None
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path

    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True

    test_data_loaders = prepare_testing_data(config)

    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    _ = test_epoch(model, test_data_loaders)
    print('===> Test Done!')


if __name__ == '__main__':
    main()
