#!/bin/bash
unit_model_folder="unit_bev_new_lyft2kitti_2channel_folder_2"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
unit_checkpoint_dir=$unit_model_dir/checkpoints/gen_00018000.pt

python analyze_mean_histogram.py --dataset lyft2kitti2 --unit_config $unit_model_dir/config.yaml --unit_checkpoint $unit_checkpoint_dir
