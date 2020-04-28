#!/bin/bash

unit_model_folder="unit_bev_new_audi2kitti_2channel_folder"
unit_model_dir="/home/user/work/master_thesis/code/UNIT/outputs/$unit_model_folder"
unit_checkpoint_dir=$unit_model_dir/checkpoints/gen_00010000.pt

conda activate ComplexYOLO_0.4.1
python perform_bev_transformation.py --dataset audi2kitti --unit_config $unit_model_dir/config.yaml --unit_checkpoint $unit_checkpoint_dir
conda activate ComplexYOLO_1.1
python eval_mAP.py --dataset audi2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth