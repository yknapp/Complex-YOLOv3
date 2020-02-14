#!/bin/bash

python test_detection.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder_11/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder_11/checkpoints/gen_00015500.pt
