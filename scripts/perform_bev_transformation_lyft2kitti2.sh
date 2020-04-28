#!/bin/bash

dataset=lyft2kitti2
num_channels=2
temp_bev_path="/home/user/work/master_thesis/datasets/lyft_kitti/object/training/bev"

# UNIT
unit_model_folder="unit_bev_lyft2kitti_2channel_folder_8"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
checkpoint="$unit_model_dir/checkpoints/gen_00018000.pt"

# flush temp BEV folder
rm $temp_bev_path/*

# transform BEV images
conda activate ComplexYOLO_0.4.1
python perform_bev_transformation.py --dataset $dataset --num_channels $num_channels --unit_config $unit_model_dir/config.yaml --unit_checkpoint $checkpoint
