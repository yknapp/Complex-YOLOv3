from __future__ import division
import os
import glob
import numpy as np
import cv2
import torch.utils.data as torch_data
import utils.dataset_utils as dataset_utils
import utils.calibration as calibration
import utils.object as object
import json

class AudiDataset(torch_data.Dataset):

    def __init__(self, split='train'):
        self.split = split

        is_test = self.split == 'test'
        root_dir = '/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes'
        self.imageset_dir = root_dir
        self.lidar_path = os.path.join(root_dir, "lidar", "cam_front_center")
        self.image_path = os.path.join(root_dir, "camera", "cam_front_center")
        self.calib_path = os.path.join(root_dir, "cams_lidars.json")
        self.label_path = os.path.join(root_dir, "label3D", "cam_front_center")

        self.CLASS_NAME_TO_ID = {
            'Car': 				    0,
            'Pedestrian': 		    1,
            'Bicycle': 			    2
        }

        if not is_test:
            split_dir = os.path.join('data', 'AUDI', 'ImageSets', split+'.txt')
            self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        else:
            self.files = sorted(glob.glob("%s/*.npz" % self.lidar_path))
            self.image_idx_list = [os.path.split(x)[1].split(".")[0].strip() for x in self.files]
            print(self.image_idx_list[0])

        self.num_samples = self.image_idx_list.__len__()

    def get_image(self, timestamp, idx):
        img_file = os.path.join(self.image_path, '%s_camera_frontcenter_%s.png' % (timestamp, idx))
        assert os.path.exists(img_file)
        return cv2.imread(img_file) # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_image_shape(self, timestamp, idx):
        img_file = os.path.join(self.image_path, '%s_camera_frontcenter_%s.png' % (timestamp, idx))
        assert os.path.exists(img_file)
        img = cv2.imread(img_file)
        width, height, channel = img.shape
        return width, height, channel

    def get_lidar(self, timestamp, idx):
        lidar_file = os.path.join(self.lidar_path, '%s_lidar_frontcenter_%s.npz' % (timestamp, idx))
        assert os.path.exists(lidar_file)
        lidar_pc_raw = np.load(lidar_file)
        lidar_pc = np.zeros([lidar_pc_raw['points'].shape[0], 4])
        lidar_pc[:, :3] = lidar_pc_raw['points']
        lidar_pc[:, 3] = lidar_pc_raw['reflectance']
        return lidar_pc

    def get_calib(self):
        calib_file = self.calib_path
        assert os.path.exists(calib_file)
        return calibration.AudiCalibration(calib_file)

    def get_label(self, timestamp, idx):
        label_file = os.path.join(self.label_path, '%s_label3D_frontcenter_%s.json' % (timestamp, idx))
        assert os.path.exists(label_file)
        with open(label_file, 'r') as f:
            bboxs = json.load(f)
        objects = [object.AudiObject3d(bboxs[bbox]) for bbox in bboxs.keys()]
        return objects

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented
