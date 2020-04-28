#!/bin/bash

dataset=lyft2kitti2
num_channel=1
file_index=488

unit_model_folder="unit_bev_new_lyft2kitti_1channel_folder"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
unit_checkpoint_dir=$unit_model_dir/checkpoints/gen_00012000.pt

python analyze_object_pixel_inspection.py --file_index $file_index --dataset $dataset --num_channel $num_channel --unit_config $unit_model_dir/config.yaml --unit_checkpoint $unit_checkpoint_dir
