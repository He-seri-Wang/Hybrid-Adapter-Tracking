import os
import cv2
import sys
from os.path import join, isdir, dirname
import numpy as np
import argparse
import multiprocessing
import torch
import time

prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.bat import BATTrack
import lib.test.parameter.bat as rgbt_adapter_params
from lib.train.dataset.depth_utils import get_x_frame


def genConfig(seq_path, set_type):
    if set_type == 'LasHeR':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    else:
        raise ValueError("Only LasHeR shown here for brevity")

    return RGB_img_list, T_img_list, RGB_gt, T_gt


class BAT_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        gt_bbox_np = np.array(region).astype(np.float32)
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        outputs = self.tracker.track(img_RGB)
        return outputs['target_bbox'], outputs['best_score'], outputs['time']


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=60):

    worker_name = multiprocessing.current_process().name
    worker_id = int(worker_name.split('-')[-1]) - 1 if '-' in worker_name else 0
    gpu_id = worker_id % num_gpu
    torch.cuda.set_device(gpu_id)

    # tracker
    params = rgbt_adapter_params.parameters(yaml_name, epoch)
    mmtrack = BATTrack(params)
    tracker = BAT_RGBT(tracker=mmtrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence:', seq_name, '——————————————')

    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)

    # ✅ 1. 预加载（去 IO）
    RGB_frames = [cv2.imread(p) for p in RGB_img_list]
    T_frames = [cv2.imread(p) for p in T_img_list]

    result = np.zeros_like(RGB_gt)
    result[0] = np.copy(RGB_gt[0])

    total_time = 0
    count = 0

    for frame_idx, (rgb, T) in enumerate(zip(RGB_frames, T_frames)):

        image = get_x_frame(rgb, T, dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))

        if frame_idx == 0:
            tracker.initialize(image, RGB_gt[0].tolist())
        else:
            region, confidence, t = tracker.track(image)

            result[frame_idx] = np.array(region)

            total_time += t
            count += 1

    fps = count / total_time if total_time > 0 else 0
    print('{} , VIT_FPS: {}'.format(seq_name, fps))


# ===========================
# 🔥 修改 BATTrack.track
# ===========================
def inject_timing_into_battrack():
    """
    monkey patch: 给 BATTrack.track 加 timing
    """
    old_track = BATTrack.track

    def new_track(self, image):

        # === 原 preprocessing ===
        outputs = {}

        # ⚠️ 关键：你需要定位 forward 那一行
        # 这里假设在 old_track 内部会调用 self.network.forward

        torch.cuda.synchronize()
        start = time.time()

        out = old_track(self, image)

        torch.cuda.synchronize()
        end = time.time()

        forward_time = end - start

        # 在原返回基础上加时间
        out['time'] = forward_time

        return out

    BATTrack.track = new_track


if __name__ == '__main__':

    # 注入 timing
    inject_timing_into_battrack()

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_name', type=str, required=True)
    parser.add_argument('--dataset_name', default='LasHeR')
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int)
    parser.add_argument('--epoch', default=60, type=int)

    args = parser.parse_args()

    dataset_name = args.dataset_name

    if dataset_name == 'LasHeR':
        seq_home = '/scratch/zceehw4/vipt/data/LasHeR/testingset'
        seq_list = sorted([f for f in os.listdir(seq_home) if isdir(join(seq_home, f))])
    else:
        raise ValueError

    start = time.time()

    sequence_list = [
        (s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch)
        for s in seq_list
    ]

    multiprocessing.set_start_method('spawn', force=True)

    with multiprocessing.Pool(processes=args.threads) as pool:
        pool.starmap(run_sequence, sequence_list)

    print("Total time:", time.time() - start)