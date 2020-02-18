#!/bin/bash

echo "14500:"
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00014500.pt

echo "15500:"
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00015500.pt

echo "34500:"
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00034500.pt

echo "37500:"
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00037500.pt

echo "56000:"
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00056000.pt

echo "80000:"
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00080000.pt

echo "80500:"
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00080500.pt

echo "84500:"
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00084500.pt

echo "100000:"
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00100000.pt

echo "101000":
python eval_mAP.py --dataset lyft2kitti --model_def config/complex_yolov3_low_res.cfg --weights_path checkpoints/test_1/weights/yolov3_ckpt_epoch-290_MAP-0.82.pth --unit_config /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/config.yaml --unit_checkpoint /home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder/checkpoints/gen_00101000.pt
