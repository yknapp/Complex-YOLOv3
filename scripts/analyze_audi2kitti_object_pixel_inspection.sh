#!/bin/bash
unit_model_folder="unit_bev_new_audi2kitti_2channel_folder"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
unit_checkpoint_dir=$unit_model_dir/checkpoints/gen_00025000.pt

python analyze_object_pixel_inspection.py --file_index 1 --dataset audi2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config $unit_model_dir/config.yaml --unit_checkpoint $unit_checkpoint_dir
