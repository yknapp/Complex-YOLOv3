#!/bin/bash

python test_detection.py --dataset lyft --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/yolov3_ckpt_epoch-142_MAP-0.79.pth
