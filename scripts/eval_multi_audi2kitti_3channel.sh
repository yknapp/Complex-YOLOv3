#!/bin/bash

dataset=audi2kitti
num_channels=3
temp_bev_path="/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/bev"

# UNIT
unit_model_folder="unit_bev_new_audi2kitti_3channel_folder"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"

# ComplexYOLO
yolo_weight=""
yolo_model="complex_yolov3_low_res.cfg"

# output
output_file="$unit_model_dir/$unit_model_folder.txt"

"" > $output_file
for checkpoint in $unit_model_dir/checkpoints/"gen_"*".pt"
do
  rm $temp_bev_path/*

  echo "CHECKPOINT: $checkpoint" >> $output_file  
  
  # create BEV transformations  
  conda activate ComplexYOLO_0.4.1
  python perform_bev_transformation.py --dataset $dataset --num_channels $num_channels --unit_config $unit_model_dir/config.yaml --unit_checkpoint $checkpoint

  # evaluate transformations
  conda activate ComplexYOLO_1.1
  python eval_mAP.py --dataset $dataset --num_channels $num_channels --model_def config/$yolo_model --weights_path checkpoints/$yolo_weight >> $output_file
done
