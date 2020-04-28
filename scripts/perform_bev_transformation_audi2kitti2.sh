#!/bin/bash

dataset=audi2kitti
num_channels=2
bev_temp_path="/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/bev"

# UNIT
unit_model_folder="unit_bev_new_audi2kitti_2channel_folder_3"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
checkpoint="$unit_model_dir/checkpoints/gen_00120000.pt"

# flush temp BEV folder
rm /home/user/work/master_thesis/datasets/bev_images/lyft2kitti/*

# transform BEV images
conda activate ComplexYOLO_0.4.1
python perform_bev_transformation.py --dataset $dataset --num_channels $num_channels --unit_config $unit_model_dir/config.yaml --unit_checkpoint $checkpoint
