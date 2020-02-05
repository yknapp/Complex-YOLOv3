#!/bin/bash

COMPLEXYOLO_CONFIG="config/complex_yolov3_low_res.cfg"
COMPLEXYOLO_CHECKPOINT="checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth"
UNIT_MODEL_PATH="/home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder_10/"

for ((i=13000; i<=17000; i+=100))
do
   filename=$(printf "%08d" $i)
   echo "${filename}:"

   python eval_mAP.py --model_def "${COMPLEXYOLO_CONFIG}" --weights_path "${COMPLEXYOLO_CHECKPOINT}" --unit_config "${UNIT_MODEL_PATH}config.yaml" --unit_checkpoint "${UNIT_MODEL_PATH}checkpoints/gen_${filename}.pt"
done
