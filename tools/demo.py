# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Jiang Du (https://github.com/jiang-du)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import get_model_summary
#from utils.vis import save_debug_images
from utils.vis import save_demo_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size

import cv2

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    """
    parser.add_argument('--source',
                        help="Image or video",
                        default="test.jpg",
                        nargs=1)
    """

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    print("Initilaized.")

    if cfg.TEST.MODEL_FILE:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        raise Exception("No weight file. Would you like to test with your hammer?")

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    # data_loader, test_dataset = make_test_dataloader(cfg)

    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        # 默认是用这种
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    parser = HeatmapParser(cfg)

    print("Load model successfully.")

    img_name = "./test.jpg"
    video_name = "./IMG_0116.MOV"
    ENABLE_CAMERA = 1
    ENABLE_VIDEO = 1
    VIDEO_ROTATE = 1
    
    if ENABLE_CAMERA:
        # 读取视频流
        cap = cv2.VideoCapture(-1)
        ret, image = cap.read()
        x, y = image.shape[0:2]
        print((x, y))
        # 创建视频文件
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter('./result.mp4', fourcc, 24, (x, y), True)
        while ret:
            ret, image = cap.read()
            if not ret:
                break
            # 实时视频自动禁用scale search
            base_size, center, scale = get_multi_scale_size(
                image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
            )
            with torch.no_grad():
                final_heatmaps = None
                tags_list = []
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, 1.0, 1.0
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, 1.0, final_heatmaps, tags_list, heatmaps, tags
                )

                final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
                tags = torch.cat(tags_list, dim=4)
                grouped, scores = parser.parse(
                    final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
                )

                final_results = get_final_preds(
                    grouped, center, scale,
                    [final_heatmaps.size(3), final_heatmaps.size(2)]
                )

            detection = save_demo_image(image, final_results, mode=1)

            detection = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)
            cv2.imshow("Pose Estimation", detection)
            out.write(detection)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    elif ENABLE_VIDEO:
        # 读取视频流
        cap = cv2.VideoCapture(video_name)
        # 创建视频文件 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./result.mp4', fourcc, 24, (540, 960), True)
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            if VIDEO_ROTATE:
                image = cv2.resize(image, (960, 540)).transpose((1, 0, 2))
            # 实时视频自动禁用scale search
            base_size, center, scale = get_multi_scale_size(
                image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
            )
            with torch.no_grad():
                final_heatmaps = None
                tags_list = []
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, 1.0, 1.0
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, 1.0, final_heatmaps, tags_list, heatmaps, tags
                )

                final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
                tags = torch.cat(tags_list, dim=4)
                grouped, scores = parser.parse(
                    final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
                )

                final_results = get_final_preds(
                    grouped, center, scale,
                    [final_heatmaps.size(3), final_heatmaps.size(2)]
                )

            detection = save_demo_image(image, final_results, mode=1)

            detection = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)
            cv2.imshow("Pose Estimation", detection)
            out.write(detection)
            cv2.waitKey(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        images = cv2.imread(img_name)
        image = images
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            print(cfg.TEST.SCALE_FACTOR)
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        save_demo_image(image, final_results, file_name="./result.jpg")
        # save_debug_images(cfg, image_resized, None, None, outputs, prefix)

if __name__ == '__main__':
    main()
