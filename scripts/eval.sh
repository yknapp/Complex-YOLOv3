#!/bin/bash

python eval_mAP.py --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/yolov3_ckpt_epoch-290_MAP-0.82.pth
