#!/bin/bash

unit_model_folder="unit_bev_new_lyft2kitti_1channel_folder"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
output_file="$unit_model_dir/$unit_model_folder.txt"
"" > $output_file
for checkpoint in $unit_model_dir/checkpoints/"gen_"*".pt"
do
  rm /home/user/work/master_thesis/datasets/lyft_kitti/object/training/bev/*

  echo "CHECKPOINT: $checkpoint" >> $output_file  
  
  conda activate ComplexYOLO_0.4.1

  python perform_bev_transformation.py --dataset lyft2kitti2 --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/yolov3_ckpt_epoch-142_MAP-0.79.pth --unit_config $unit_model_dir/config.yaml --unit_checkpoint $checkpoint

  conda activate ComplexYOLO_1.1
  
  python eval_mAP.py --dataset lyft2kitti2 --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/yolov3_ckpt_epoch-142_MAP-0.79.pth >> $output_file
done
