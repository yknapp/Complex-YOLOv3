from __future__ import division
import os
import glob
import numpy as np
import cv2
import torch.utils.data as torch_data
import utils.dataset_utils as dataset_utils
import utils.calibration as calibration
import utils.object as object

class Lyft2KittiDataset(torch_data.Dataset):

    def __init__(self, split='train', folder='training'):
        self.split = split

        is_test = self.split == 'test'
        root_dir = '/home/user/work/master_thesis/datasets/lyft_kitti'
        self.imageset_dir = os.path.join(root_dir, 'object', folder)
        self.lidar_path = os.path.join(self.imageset_dir, "velodyne")

        self.image_path = os.path.join(self.imageset_dir, "image_2")
        self.calib_path = os.path.join(self.imageset_dir, "calib")
        self.label_path = os.path.join(self.imageset_dir, "label_2")
        self.bev_path =   os.path.join(self.imageset_dir, "bev")

        self.CLASS_NAME_TO_ID = {
            'car': 				    0,
            'pedestrian': 		    1,
            'bicycle': 			    2
        }

        if not is_test:
            split_dir = os.path.join('data', 'LYFT', 'ImageSets', split+'.txt')
            self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        else:
            self.files = sorted(glob.glob("%s/*.bin" % self.lidar_path))
            self.image_idx_list = [os.path.split(x)[1].split(".")[0].strip() for x in self.files]
            print(self.image_idx_list[0])

        self.num_samples = self.image_idx_list.__len__()

    def get_image(self, idx):
        img_file = os.path.join(self.image_path, '%s.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file) # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_path, '%s.png' % idx)
        assert  os.path.exists(img_file)
        img = cv2.imread(img_file)
        width, height, channel = img.shape
        return width, height, channel

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_path, '%s.bin' % idx)
        assert os.path.exists(lidar_file)
        lidar_pc = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        lidar_pc[:, 3] = 0.0  # set to zero, since lyft has no intensity values
        return lidar_pc

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, '%s.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.KittiCalibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_path, '%s.txt' % idx)
        assert os.path.exists(label_file)
        lines = [line.rstrip() for line in open(label_file)]
        objects = [object.KittiObject3d(line) for line in lines]
        return objects

    def get_bev(self, idx):
        bev_file = os.path.join(self.bev_path, '%s.npy' % idx)
        assert os.path.exists(bev_file)
        return np.load(bev_file)

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented