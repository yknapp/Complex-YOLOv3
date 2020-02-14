import os
import numpy as np
import utils.config as cnf
import utils.dataset_bev_utils as bev_utils

from unit.lyft2kitti_converter import Lyft2KittiConverter

UNIT_CONFIG_PATH = '/home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder_8/config.yaml'
UNIT_CHECKPOINT_PATH = '/home/user/work/master_thesis/code/UNIT/outputs/unit_bev_lyft2kitti_2channel_folder_8/checkpoints/gen_00010000.pt'
CHOSEN_EVAL_FILES_PATH = 'data/LYFT/ImageSets/valid.txt'
BEV_OUTPUT_PATH = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/bev'
LYFT_LIDAR_PATH = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/velodyne'

def get_lidar_lyft(idx):
    lidar_file = os.path.join(LYFT_LIDAR_PATH, '%s.bin' % idx)
    assert os.path.exists(lidar_file)
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


def perform_img2img_translation(lyft2kitti_conv, np_img):
    c, height, width = np_img.shape
    np_img_input = np.zeros((width, width, 2))
    np_img_input[:, :, 0] = np_img[2, :, :]
    np_img_input[:, :, 1] = np_img[1, :, :]
    np_img_transformed = lyft2kitti_conv.transform(np_img_input)
    np_img_output = np.zeros((3, width, width))
    np_img_output[2, :, :] = np_img_transformed[0, :, :]
    np_img_output[1, :, :] = np_img_transformed[1, :, :]
    return np_img_output

def main():
    lyft2kitti_conv = Lyft2KittiConverter(UNIT_CONFIG_PATH, UNIT_CHECKPOINT_PATH)

    # get validation images which are chosen for evaluation
    image_filename_list = [x.strip() for x in open(CHOSEN_EVAL_FILES_PATH).readlines()]

    # transform lyft2kitti
    for image_filename in image_filename_list:
        print("Processing: ", image_filename)
        if os.path.exists(os.path.join(BEV_OUTPUT_PATH, image_filename+'.npy')):
            print("File '%s' already exists" % image_filename+'.npy')
            continue
        lidarData = get_lidar_lyft(image_filename)
        b = bev_utils.removePoints(lidarData, cnf.boundary)
        bev_array = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
        bev_array_transformed = perform_img2img_translation(lyft2kitti_conv, bev_array)
        output_path = os.path.join(BEV_OUTPUT_PATH, image_filename)
        np.save('%s.npy' % output_path, bev_array_transformed)

if __name__ == '__main__':
    main()
