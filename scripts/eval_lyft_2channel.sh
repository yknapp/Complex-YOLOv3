#!/bin/bash

dataset=lyft
num_channels=2
yolo_model="complex_yolov3_low_res.cfg"
yolo_weight="bev_2channel/weights/yolov3_ckpt_epoch-260_MAP-0.81.pth"

conda activate ComplexYOLO_1.1
python eval_mAP.py --dataset $dataset --num_channels $num_channels --model_def config/$yolo_model --weights_path checkpoints/$yolo_weight
