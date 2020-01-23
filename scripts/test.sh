#!/bin/bash

python test_detection.py --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_0/yolov3_ckpt_epoch-36_MAP-0.72.pth
