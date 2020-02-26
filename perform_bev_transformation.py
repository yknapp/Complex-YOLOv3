import os
import sys
import argparse
import numpy as np
import utils.config as cnf
import utils.dataset_bev_utils as bev_utils

from unit.unit_converter import UnitConverter


def get_lidar_lyft(idx):
    lyft_lidar_path = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/velodyne'
    lidar_file = os.path.join(lyft_lidar_path, '%s.bin' % idx)
    assert os.path.exists(lidar_file)
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


def get_lidar_audi(img_filename):
    audi_lidar_path = '/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/lidar/cam_front_center'
    # extract timestamp and index out of given filename
    file_name_list = img_filename.split('_')
    timestamp = file_name_list[0]
    idx = file_name_list[3]
    lidar_file = os.path.join(audi_lidar_path, '%s_lidar_frontcenter_%s.npz' % (timestamp, idx))
    assert os.path.exists(lidar_file)
    lidar_pc_raw = np.load(lidar_file)
    lidar_pc = np.zeros([lidar_pc_raw['points'].shape[0], 4])
    lidar_pc[:, :3] = lidar_pc_raw['points']
    lidar_pc[:, 3] = lidar_pc_raw['reflectance']
    return lidar_pc


def get_dataset_info(dataset_name):
    if dataset_name == 'lyft2kitti2':
        chosen_eval_files_path = 'data/LYFT/ImageSets/valid.txt'
        bev_output_path = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/bev'
        get_lidar = get_lidar_lyft
    elif dataset_name == 'audi2kitti':
        chosen_eval_files_path = 'data/AUDI/ImageSets/valid.txt'
        bev_output_path = '/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/bev'
        get_lidar = get_lidar_audi
    else:
        print("Unknown dataset '%s'" % dataset_name)
        sys.exit()
    return chosen_eval_files_path, bev_output_path, get_lidar


def perform_img2img_translation(lyft2kitti_conv, np_img):
    c, height, width = np_img.shape
    np_img_input = np.zeros((width, width, 3))
    np_img_input[:, :, 0] = np_img[2, :, :]
    np_img_input[:, :, 1] = np_img[1, :, :]
    np_img_input[:, :, 2] = np_img[0, :, :]
    np_img_transformed = lyft2kitti_conv.transform(np_img_input)
    np_img_output = np.zeros((3, width, width))
    np_img_output[2, :, :] = np_img_transformed[0, :, :]
    np_img_output[1, :, :] = np_img_transformed[1, :, :]
    return np_img_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="None", help="chose dataset (lyft2kitti2, audi2kitti)")
    parser.add_argument("--model_def", type=str, default="config/complex_tiny_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/tiny-yolov3_ckpt_epoch-220.pth", help="path to weights file")
    parser.add_argument('--unit_config', type=str, default=None, help="UNIT net configuration")
    parser.add_argument('--unit_checkpoint', type=str, default=None, help="checkpoint of UNIT autoencoders")
    opt = parser.parse_args()

    # get specific information to chosen dataset
    chosen_eval_files_path, bev_output_path, get_lidar = get_dataset_info(opt.dataset)

    unit_conv = UnitConverter(opt.unit_config, opt.unit_checkpoint)

    # get validation images which are chosen for evaluation
    image_filename_list = [x.strip() for x in open(chosen_eval_files_path).readlines()]

    # transform every pointcloud to bev and transform with unit
    for image_filename in image_filename_list:
        print("Processing: ", image_filename)
        #if os.path.exists(os.path.join(bev_output_path, image_filename+'.npy')):
        #    print("File '%s' already exists" % image_filename+'.npy')
        #    continue
        lidarData = get_lidar(image_filename)
        b = bev_utils.removePoints(lidarData, cnf.boundary)
        bev_array = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
        bev_array_transformed = perform_img2img_translation(unit_conv, bev_array)
        output_path = os.path.join(bev_output_path, image_filename)
        np.save('%s.npy' % output_path, bev_array_transformed)


if __name__ == '__main__':
    main()
