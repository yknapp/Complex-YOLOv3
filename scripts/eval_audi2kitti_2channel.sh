#!/bin/bash

dataset=audi2kitti
num_channels=2

# UNIT
unit_model_folder="unit_bev_new_audi2kitti_2channel_folder_3"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
unit_checkpoint_dir=$unit_model_dir/checkpoints/gen_00151500.pt

# ComplexYOLO
yolo_weight="bev_2channel/weights/yolov3_ckpt_epoch-260_MAP-0.81.pth"
yolo_model="complex_yolov3_low_res.cfg"

# create BEV transformations
conda activate ComplexYOLO_0.4.1
python perform_bev_transformation.py --dataset $dataset --num_channels $num_channels --unit_config $unit_model_dir/config.yaml --unit_checkpoint $unit_checkpoint_dir

# evaluate transformations
conda activate ComplexYOLO_1.1
python eval_mAP.py --dataset $dataset --num_channels $num_channels --model_def config/$yolo_model --weights_path checkpoints/$yolo_weight
