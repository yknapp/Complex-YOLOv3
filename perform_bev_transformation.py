import os
import sys
import argparse
import numpy as np
import utils.config as cnf
import utils.dataset_bev_utils as bev_utils
import postprocessing
import imageio

from unit.unit_converter import UnitConverter


def get_lidar_lyft(idx):
    lyft_lidar_path = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/velodyne'
    lidar_file = os.path.join(lyft_lidar_path, '%s.bin' % idx)
    assert os.path.exists(lidar_file)
    lidar_pc = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    lidar_pc[:, 3] = 0.0  # intensity channel of Lyft pointcloud is always 100, so erase this
    return lidar_pc


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
        #bev_output_path = '/home/user/work/master_thesis/datasets/bev_images/lyft2kitti'
        get_lidar = get_lidar_lyft
    elif dataset_name == 'audi2kitti':
        chosen_eval_files_path = 'data/AUDI/ImageSets/valid.txt'
        bev_output_path = '/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/bev'
        #bev_output_path = '/home/user/work/master_thesis/datasets/bev_images/audi2kitti'
        get_lidar = get_lidar_audi
    else:
        print("Unknown dataset '%s'" % dataset_name)
        sys.exit()
    return chosen_eval_files_path, bev_output_path, get_lidar


def shift_image(input_img, x_shift=0, y_shift=0):
    input_img_shifted = np.copy(input_img)
    input_img_shifted = np.roll(input_img_shifted, x_shift)
    input_img_shifted = np.roll(input_img_shifted, y_shift, axis=1)
    # clear the pixels, which were shifted from the right to the left side of the image
    if x_shift < 0:
        input_img_shifted[:, :, x_shift:] = 0.0
    else:
        input_img_shifted[:, :, :x_shift] = 0.0
    if y_shift < 0:
        input_img_shifted[:, y_shift:, :] = 0.0
    else:
        input_img_shifted[:, :y_shift, :] = 0.0
    return input_img_shifted


def perform_img2img_translation(lyft2kitti_conv, np_img_input, num_channel):
    np_img = np.copy(np_img_input)
    c, height, width = np_img.shape
    if num_channel == 1:
        np_img_input1 = np.zeros((width, width, 1))
        np_img_input1[:, :, 0] = np_img[0, :, :]  # height
        print("IMG2IMG TRANSLATION: 1 Channel")
    elif num_channel == 2:
        np_img_input = np.zeros((width, width, 2))
        np_img_input[:, :, 0] = np_img[2, :, :]  # density
        np_img_input[:, :, 1] = np_img[1, :, :]  # height
        print("IMG2IMG TRANSLATION: 2 Channels")
    elif num_channel == 3:
        np_img_input = np.zeros((width, width, 3))
        np_img_input[:, :, 0] = np_img[2, :, :]  # density
        np_img_input[:, :, 1] = np_img[1, :, :]  # height
        np_img_input[:, :, 2] = np_img[0, :, :]  # intensity
        print("IMG2IMG TRANSLATION: 3 Channels")
    np_img_transformed = lyft2kitti_conv.transform(np_img_input)
    # add shift to compensate the shift of UNIT transformation
    #np_img_transformed = shift_image(np_img_transformed, x_shift=-6, y_shift=1)
    #np_img_transformed = shift_image(np_img_transformed, x_shift=1, y_shift=-2)
    np_img_output = np.zeros((3, width, width))
    np_img_output[2, :, :] = np_img_transformed[0, :, :]  # density
    np_img_output[1, :, :] = np_img_transformed[1, :, :]  # height
    if num_channel == 3:
        np_img_output[0, :, :] = np_img_transformed[2, :, :]  # intensity
        print("IMG2IMG TRANSLATION OUTPUT: 3 Channels")
    elif num_channel == 1:
        print("IMG2IMG TRANSLATION OUTPUT: 1 Channel")
        np_img_output[0, :, :] = np_img_transformed[0, :, :]  # height
        np_img_output[1, :, :] = np_img_transformed[0, :, :]  # height
        np_img_output[2, :, :] = np_img_transformed[0, :, :]  # height
    else:
        print("IMG2IMG TRANSLATION OUTPUT: 2 Channels")
    return np_img_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="None", help="choose dataset (lyft2kitti2, audi2kitti)")
    parser.add_argument("--num_channel", type=int, default=None, help="Number of channels")
    parser.add_argument("--model_def", type=str, default="config/complex_tiny_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/tiny-yolov3_ckpt_epoch-220.pth", help="path to weights file")
    parser.add_argument('--unit_config', type=str, default=None, help="UNIT net configuration")
    parser.add_argument('--unit_checkpoint', type=str, default=None, help="checkpoint of UNIT autoencoders")
    opt = parser.parse_args()

    if not opt.num_channel:
        print("Error: Select number of channels!")
        exit()

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
        bev_array = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary, opt.num_channel)
        bev_array_transformed = perform_img2img_translation(unit_conv, bev_array, opt.num_channel)
        #bev_array_transformed = postprocessing.blacken_pixel(bev_array_transformed, threshold=0.2)
        #bev_array_transformed[:, :, 2] = postprocessing.blacken_pixel(bev_array_transformed[:, :, 2], threshold=0.2)  # density
        #bev_array_transformed[:, :, 1] = postprocessing.blacken_pixel(bev_array_transformed[:, :, 1], threshold=0.2)  # height
        output_path = os.path.join(bev_output_path, image_filename)

        # save as numpy
        np.save('%s.npy' % output_path, bev_array_transformed)

        # save as image
        # transform from ComplexYOLO image style (3, x, y) to normal style (x, y, 3)
        #bev_array_transformed_img = np.zeros((bev_array_transformed.shape[1], bev_array_transformed.shape[2], 3))
        #bev_array_transformed_img[:, :, 2] = bev_array_transformed[0, :, :]  # density
        #bev_array_transformed_img[:, :, 1] = bev_array_transformed[1, :, :]  # height
        #bev_array_transformed_img[:, :, 0] = bev_array_transformed[2, :, :]  # intensity
        #imageio.imwrite('%s.png' % output_path, bev_array_transformed_img)


if __name__ == '__main__':
    main()
