from __future__ import division

import numpy as np
import cv2
import torch
import sys

import utils.dataset_bev_utils as bev_utils
from utils.kitti_yolo_dataset import KittiYOLODataset
from utils.lyft_yolo_dataset import LyftYOLODataset
from utils.lyft2kitti_yolo_dataset import Lyft2KittiYOLODataset
from torch.utils.data import DataLoader
import utils.config as cnf

if __name__ == "__main__":
    dataset = 'lyft2kitti'  # kitti, lyft or lyft2kitti

    # unit files for lyft2kitti
    unit_config = '/home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder_8/config.yaml'
    unit_checkpoint = '/home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder_8/checkpoints/gen_00010000.pt'

    img_size=cnf.BEV_WIDTH

    # Get dataloader
    if dataset == 'kitti':
        dataset = KittiYOLODataset(
            split='valid',
            mode='TRAIN',
            folder='training',
            data_aug=True,
        )
    elif dataset == 'lyft':
        dataset = LyftYOLODataset(
            split='valid',
            mode='TRAIN',
            folder='training',
            data_aug=True,
        )
    elif dataset == 'lyft2kitti':
        dataset = Lyft2KittiYOLODataset(
            unit_config_path=unit_config,
            unit_checkpoint_path=unit_checkpoint,
            split='valid',
            mode='TRAIN',
            folder='training',
            data_aug=True,
        )
    else:
        print("Unknown dataset '%s'" % dataset)
        sys.exit()

    # Load Dataset
    dataloader = DataLoader(
        dataset,
        1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        # Rescale target
        targets[:, 2:6] *= img_size
        # Get yaw angle
        targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])

        img = imgs.squeeze() * 255
        img = img.permute(1,2,0).numpy().astype(np.uint8)        
        img_display = np.zeros((img_size, img_size, 3), np.uint8)
        img_display[...] = img[...]
        
        for c,x,y,w,l,yaw in targets[:, 1:7].numpy():
            # Draw rotated box
            bev_utils.drawRotatedBox(img_display, x, y, w, l, yaw, cnf.colors[int(c)])

        cv2.imshow('img-kitti-bev', img_display)

        if cv2.waitKey(0) & 0xff == 27:
            break