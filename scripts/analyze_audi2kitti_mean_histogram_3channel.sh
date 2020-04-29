#!/bin/bash

dataset=audi2kitti
num_channel=3

unit_model_folder="unit_bev_new_audi2kitti_3channel_folder"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
unit_checkpoint_dir=$unit_model_dir/checkpoints/gen_00037500.pt

python analyze_mean_histogram.py --dataset $dataset --num_channel $num_channel --unit_config $unit_model_dir/config.yaml --unit_checkpoint $unit_checkpoint_dir
