#!/bin/bash

dataset=lyft2kitti2
num_channels=3

# UNIT
unit_model_folder="unit_bev_new_lyft2kitti_3channel_folder"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
unit_checkpoint_dir=$unit_model_dir/checkpoints/

# ComplexYOLO
yolo_weight=""
yolo_model="complex_yolov3_low_res.cfg"

# create BEV transformations
conda activate ComplexYOLO_0.4.1
python perform_bev_transformation.py --dataset $dataset --num_channels $num_channels --unit_config $unit_model_dir/config.yaml --unit_checkpoint $unit_checkpoint_dir

# evaluate transformations
conda activate ComplexYOLO_1.1
python eval_mAP.py --dataset $dataset --num_channels $num_channels --model_def config/$yolo_model --weights_path checkpoints/$yolo_weight
