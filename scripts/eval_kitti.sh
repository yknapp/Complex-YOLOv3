#!/bin/bash

conda activate ComplexYOLO_1.1
python eval_mAP.py --dataset kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth
