import os
import sys
import numpy as np
import argparse
import time
from os.path import join, dirname

prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.bat import BATTrack
import lib.test.parameter.bat as rgbt_adapter_params
import torch
from lib.train.dataset.depth_utils import get_x_frame


# =========================
# CUDA config
# =========================
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================
# dataset loader
# =========================
def genConfig(seq_path, set_type):
    if set_type == 'LasHeR':
        RGB_img_list = sorted([
            seq_path + '/visible/' + p
            for p in os.listdir(seq_path + '/visible')
            if p.endswith(".jpg")
        ])

        T_img_list = sorted([
            seq_path + '/infrared/' + p
            for p in os.listdir(seq_path + '/infrared')
            if p.endswith(".jpg")
        ])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    else:
        raise ValueError("Only LasHeR supported")

    return RGB_img_list, T_img_list, RGB_gt, T_gt


# =========================
# wrapper
# =========================
class BAT_RGBT(object):
    def __init__(self, tracker, use_amp=True):
        self.tracker = tracker
        self.use_amp = use_amp

    def initialize(self, image, region):
        self.tracker.initialize(image, {'init_bbox': list(region)})

    def track(self, img):
        if self.use_amp:
            with torch.cuda.amp.autocast():
                out = self.tracker.track(img)
        else:
            out = self.tracker.track(img)

        return out['target_bbox'], out['best_score']


# =========================
# 🔥 STEP 1: preload FULL DATASET INTO RAM
# （完全消除网盘/磁盘影响）
# =========================
def preload_dataset(RGB_list, T_list, dtype):
    rgb_cache = []
    t_cache = []

    print("[INFO] Preloading dataset into RAM...")

    for i in range(len(RGB_list)):
        img = get_x_frame(RGB_list[i], T_list[i], dtype=dtype)

        rgb_cache.append(img)

    print("[INFO] Preload done.")
    return rgb_cache


# =========================
# RUNNER (benchmark version)
# =========================
def run_sequence(seq_name, seq_home, dataset_name, yaml_name, epoch):
    seq_path = os.path.join(seq_home, seq_name)

    print(f"\nProcessing: {seq_name}")

    params = rgbt_adapter_params.parameters(yaml_name, epoch)
    tracker = BAT_RGBT(BATTrack(params), use_amp=True)

    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)

    dtype = getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb')

    # =========================
    # STEP 2: preload (NO IO in benchmark)
    # =========================
    cache = preload_dataset(RGB_img_list, T_img_list, dtype)

    # =========================
    # init (NOT counted)
    # =========================
    init_img = cache[0]
    tracker.initialize(init_img, RGB_gt[0].tolist())

    # warmup (NOT counted)
    for _ in range(10):
        tracker.track(cache[1])

    torch.cuda.synchronize()

    # =========================
    # STEP 3: PURE GPU TIMING
    # =========================
    frames = 0

    torch.cuda.synchronize()
    start = time.perf_counter()

    for i in range(1, len(cache)):
        bbox, score = tracker.track(cache[i])
        frames += 1

    torch.cuda.synchronize()
    end = time.perf_counter()

    fps = frames / (end - start)

    print(f"[{seq_name}] FPS: {fps:.3f} | Frames: {frames}")

    return fps


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='LasHeR')
    parser.add_argument('--yaml_name', type=str, default='rgbt')
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--video', type=str, required=True)

    args = parser.parse_args()

    seq_home = '/scratch/zceehw4/vipt/data/LasHeR/testingset'

    run_sequence(
        args.video,
        seq_home,
        args.dataset_name,
        args.yaml_name,
        args.epoch
    )