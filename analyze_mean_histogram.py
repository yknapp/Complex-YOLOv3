import os
import numpy as np
import cv2
import glob
import imageio
import matplotlib.pyplot as plt
import argparse
import utils.config as cnf
import utils.dataset_bev_utils as bev_utils

from skimage.measure import compare_ssim
from unit.unit_converter import UnitConverter


def get_lidar_kitti(kitti_dataset, filename):
    lidar = kitti_dataset.get_lidar(int(filename))
    return lidar


def get_lidar_lyft2kitti(lyft2kitti_dataset, filename):
    lidar = lyft2kitti_dataset.get_lidar(filename)
    return lidar


def get_lidar_audi2kitti(audi2kitti_dataset, filename):
    # extract timestamp and index out of filename
    filename_list = filename.split('_')
    timestamp = filename_list[0]
    idx = filename_list[3]
    # fetch lidar
    lidar = audi2kitti_dataset.get_lidar(timestamp, idx)
    return lidar


def get_dataset_info(dataset):
    if dataset == 'kitti':
        dataset_name = 'KITTI'
        chosen_eval_files_path = 'data/KITTI/ImageSets/valid.txt'
        from utils.kitti_dataset import KittiDataset
        dataset = KittiDataset()
        get_lidar = get_lidar_kitti
    elif dataset == 'lyft2kitti2':
        dataset_name = 'Lyft'
        chosen_eval_files_path = 'data/LYFT/ImageSets/valid.txt'
        from utils.lyft2kitti_dataset2 import Lyft2KittiDataset
        dataset = Lyft2KittiDataset()
        get_lidar = get_lidar_lyft2kitti
    elif dataset == 'audi2kitti':
        dataset_name = 'Audi'
        chosen_eval_files_path = 'data/AUDI/ImageSets/valid.txt'
        from utils.audi2kitti_dataset import Audi2KittiDataset
        dataset = Audi2KittiDataset()
        get_lidar = get_lidar_audi2kitti
    else:
        print("Unknown dataset '%s'" % dataset)
        sys.exit()
    return dataset, dataset_name, chosen_eval_files_path, get_lidar


def perform_img2img_translation(lyft2kitti_conv, np_img_input):
    np_img = np.copy(np_img_input)
    height, width, c = np_img.shape
    np_img_transformed = lyft2kitti_conv.transform(np_img)
    np_img_output = np.zeros((width, width, 2))
    np_img_output[:, :, 0] = np_img_transformed[0, :, :]
    np_img_output[:, :, 1] = np_img_transformed[1, :, :]
    return np_img_output


def calc_difference_img(image_a, image_b):
    (score, diff) = compare_ssim(image_a, image_b, full=True)
    diff_img = (diff * 255).astype("uint8")
    return diff_img


def create_histogram(image, num_bin):
    return cv2.calcHist([image], [0], None, [num_bin], [0, 255])


def plot_histogram(histogram, label, title, show=False, clear=False):
    if clear:
        # clear plot from previous data
        plt.clf()
    plt.plot(histogram, label=label)
    plt.xlim([0, histogram.shape[0]])
    plt.yscale('log')
    plt.xlabel('Grayscale value')
    plt.ylabel('Number of pixels')
    plt.title(title)
    plt.legend(loc="upper right")
    if show:
        plt.show()


def save_curr_histogram(output_filename):
    plt.savefig(output_filename, dpi=300)


def create_height_density_images(filename, pointcloud_original, pointcloud_transformed):
    # density and height for original pointcloud
    pointcloud_original_filename = filename + '_original'
    output_filename_height = os.path.join(ANALYZE_OUTPUT_PATH, '%s_transformed_height.png' % pointcloud_original_filename)
    output_filename_density = os.path.join(ANALYZE_OUTPUT_PATH, '%s_transformed_density.png' % pointcloud_original_filename)
    imageio.imwrite(output_filename_density, pointcloud_original[:, :, 0])
    imageio.imwrite(output_filename_height, pointcloud_original[:, :, 1])

    # density and height for transformed pointcloud
    pointcloud_transformed_filename = filename + '_transformed'
    output_filename_height = os.path.join(ANALYZE_OUTPUT_PATH, '%s_transformed_height.png' % pointcloud_transformed_filename)
    output_filename_density = os.path.join(ANALYZE_OUTPUT_PATH, '%s_transformed_density.png' % pointcloud_transformed_filename)
    imageio.imwrite(output_filename_density, pointcloud_transformed[:, :, 0])
    imageio.imwrite(output_filename_height, pointcloud_transformed[:, :, 1])

    # show differences as images
    diff_img_density = calc_difference_img(pointcloud_original[:, :, 0], pointcloud_transformed[:, :, 0])
    diff_img_height = calc_difference_img(pointcloud_original[:, :, 1], pointcloud_transformed[:, :, 1])
    output_filename_diff_density = os.path.join(ANALYZE_OUTPUT_PATH, '%s_diff_density.png' % filename)
    output_filename_diff_height = os.path.join(ANALYZE_OUTPUT_PATH, '%s_diff_height.png' % filename)
    imageio.imwrite(output_filename_diff_density, diff_img_density[:, :, 0])
    imageio.imwrite(output_filename_diff_height, diff_img_height[:, :, 1])


def subplot_images(pointcloud_original, pointcloud_transformed):
    f, axarr = plt.subplots(2, 2)
    f.set_figheight(15)
    f.set_figwidth(15)
    axarr[0, 0].imshow(pointcloud_original[:, :, 0])
    axarr[0, 1].imshow(pointcloud_original[:, :, 1])
    axarr[1, 0].imshow(pointcloud_transformed[:, :, 0])
    axarr[1, 1].imshow(pointcloud_transformed[:, :, 1])
    axarr[0, 0].axis('off')
    axarr[0, 1].axis('off')
    axarr[1, 0].axis('off')
    axarr[1, 1].axis('off')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="None", help="chose dataset (lyft2kitti2, audi2kitti)")
    parser.add_argument('--unit_config', type=str, default=None, help="UNIT net configuration")
    parser.add_argument('--unit_checkpoint', type=str, default=None, help="checkpoint of UNIT autoencoders")
    opt = parser.parse_args()

    # get specific information to chosen dataset
    dataset, dataset_name, chosen_eval_files_path, get_lidar = get_dataset_info(opt.dataset)

    if opt.dataset != 'kitti':
        unit_conv = UnitConverter(opt.unit_config, opt.unit_checkpoint)

    # get validation images which are chosen for evaluation
    filename_list = [x.strip() for x in open(chosen_eval_files_path).readlines()]

    if opt.dataset != 'kitti':
        hist_original_sum_height = None
        hist_transformed_sum_height = None
        hist_original_sum_density = None
        hist_transformed_sum_density = None
        number_of_files = 0
        for filename in filename_list:
            print("Processing: ", filename)

            # create bevs
            lidar = get_lidar(dataset, filename)
            b = bev_utils.removePoints(lidar, cnf.boundary)
            bev_array_raw = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            bev_array = np.zeros((bev_array_raw.shape[1], bev_array_raw.shape[2], 2))
            bev_array[:, :, 0] = bev_array_raw[2, :, :]
            bev_array[:, :, 1] = bev_array_raw[1, :, :]
            bev_array_transformed = perform_img2img_translation(unit_conv, bev_array)

            # convert from float32 to int8
            bev_original_int = (np.round_(bev_array * 255)).astype(np.uint8)
            bev_transformed_int = (np.round_(bev_array_transformed * 255)).astype(np.uint8)

            # create histograms
            hist_original_density = create_histogram(bev_original_int[:, :, 0], 255)
            hist_transformed_density = create_histogram(bev_transformed_int[:, :, 0], 255)
            hist_original_height = create_histogram(bev_original_int[:, :, 1], 255)
            hist_transformed_height = create_histogram(bev_transformed_int[:, :, 1], 255)

            # add to sum histograms
            # DENSITY
            if hist_original_sum_density is not None:
                hist_original_sum_density += hist_original_density
            else:
                hist_original_sum_density = hist_original_density
            if hist_transformed_sum_density is not None:
                hist_transformed_sum_density += hist_transformed_density
            else:
                hist_transformed_sum_density = hist_transformed_density
            # HEIGHT
            if hist_original_sum_height is not None:
                hist_original_sum_height += hist_original_height
            else:
                hist_original_sum_height = hist_original_height
            if hist_transformed_sum_height is not None:
                hist_transformed_sum_height += hist_transformed_height
            else:
                hist_transformed_sum_height = hist_transformed_height

            # count number of files
            number_of_files += 1

        # calculate mean histogram
        # DENSITY
        hist_original_mean_density = np.true_divide(hist_original_sum_density, number_of_files)
        hist_transformed_mean_density = np.true_divide(hist_transformed_sum_density, number_of_files)
        plot_histogram(hist_original_mean_density, label=dataset_name, title='BEV Height Histogram Density')
        plot_histogram(hist_transformed_mean_density, label=dataset_name + '2kitti', title='BEV Height Histogram Density')
        save_curr_histogram(output_filename='mean_density_histogram')
        # HEIGHT
        hist_original_mean_height = np.true_divide(hist_original_sum_height, number_of_files)
        hist_transformed_mean_height = np.true_divide(hist_transformed_sum_height, number_of_files)
        plot_histogram(hist_original_mean_height, label=dataset_name, title='BEV Height Histogram Height', clear=True)
        plot_histogram(hist_transformed_mean_height, label=dataset_name+'2kitti', title='BEV Height Histogram Height')
        save_curr_histogram(output_filename='mean_height_histogram')

    # KITTI without transformations
    else:
        hist_sum_height = None
        hist_sum_density = None
        number_of_files = 0
        for filename in filename_list:
            print("Processing: ", filename)

            # create bevs
            lidar = get_lidar(dataset, filename)
            b = bev_utils.removePoints(lidar, cnf.boundary)
            bev_array_raw = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            bev_array = np.zeros((bev_array_raw.shape[1], bev_array_raw.shape[2], 2))
            bev_array[:, :, 0] = bev_array_raw[2, :, :]
            bev_array[:, :, 1] = bev_array_raw[1, :, :]

            # convert from float32 to int8
            bev_int = (np.round_(bev_array * 255)).astype(np.uint8)

            # create histograms
            hist_density = create_histogram(bev_int[:, :, 0], 255)
            hist_height = create_histogram(bev_int[:, :, 1], 255)

            # add to sum histograms
            # DENSITY
            if hist_sum_density is not None:
                hist_sum_density += hist_density
            else:
                hist_sum_density = hist_density
            # HEIGHT
            if hist_sum_height is not None:
                hist_sum_height += hist_height
            else:
                hist_sum_height = hist_height

            # count number of files
            number_of_files += 1

        # calculate mean histogram
        # DENSITY
        hist_original_mean_density = np.true_divide(hist_sum_density, number_of_files)
        plot_histogram(hist_original_mean_density, label=dataset_name, title='BEV Height Histogram Density')
        save_curr_histogram(output_filename='mean_density_histogram')
        # HEIGHT
        hist_original_mean_height = np.true_divide(hist_sum_height, number_of_files)
        plot_histogram(hist_original_mean_height, label=dataset_name, title='BEV Height Histogram Height', clear=True)
        save_curr_histogram(output_filename='mean_height_histogram')


if __name__ == '__main__':
    main()
