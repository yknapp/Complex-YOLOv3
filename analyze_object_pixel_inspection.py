import sys
import imageio
import argparse
import numpy as np
import utils.config as cnf
import utils.dataset_bev_utils as bev_utils
import utils.dataset_aug_utils as aug_utils
import matplotlib.pyplot as plt
import postprocessing

from matplotlib import cm
from matplotlib import colors
from unit.unit_converter import UnitConverter


def get_lidar_label_calib_lyft2kitti(lyft2kitti_dataset, filename):
    lidar = lyft2kitti_dataset.get_lidar(filename)
    labels = lyft2kitti_dataset.get_label(filename)
    calib = lyft2kitti_dataset.get_calib(filename)
    return lidar, labels, calib


def get_lidar_label_calib_audi2kitti(audi2kitti_dataset, filename):
    # extract timestamp and index out of filename
    filename_list = filename.split('_')
    timestamp = filename_list[0]
    idx = filename_list[3]
    # fetch data
    lidar = audi2kitti_dataset.get_lidar(timestamp, idx)
    labels = audi2kitti_dataset.get_label(timestamp, idx)
    calib = audi2kitti_dataset.get_calib()
    return lidar, labels, calib


def get_dataset_info(dataset):
    if dataset == 'lyft2kitti2':
        dataset_name = 'Lyft'
        chosen_eval_files_path = 'data/LYFT/ImageSets/valid.txt'
        from utils.lyft2kitti_dataset2 import Lyft2KittiDataset
        dataset = Lyft2KittiDataset()
        get_lidar_label_calib = get_lidar_label_calib_lyft2kitti
    elif dataset == 'audi2kitti':
        dataset_name = 'Audi'
        chosen_eval_files_path = 'data/AUDI/ImageSets/valid.txt'
        from utils.audi2kitti_dataset import Audi2KittiDataset
        dataset = Audi2KittiDataset()
        get_lidar_label_calib = get_lidar_label_calib_audi2kitti
    else:
        print("Unknown dataset '%s'" % dataset)
        sys.exit()
    return dataset, dataset_name, chosen_eval_files_path, get_lidar_label_calib


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


def perform_img2img_translation(lyft2kitti_conv, np_img_input, num_channels):
    height, width, c = np_img_input.shape
    if num_channels == 1:
        np_img = np.zeros((width, width, 1))
        print("1: ", np_img.shape)
        print("2: ", np_img_input.shape)
        np_img[:, :, 0] = np_img_input[:, :, 0]  # height 1 channel
    elif num_channels == 2:
        np_img = np.zeros((width, width, 2))
        np_img[:, :, 0] = np_img_input[:, :, 2]  # density
        np_img[:, :, 1] = np_img_input[:, :, 1]  # height
    elif num_channels == 3:
        np_img = np.zeros((width, width, 3))
        np_img[:, :, 0] = np_img_input[:, :, 2]  # density
        np_img[:, :, 1] = np_img_input[:, :, 1]  # height
        np_img[:, :, 2] = np_img_input[:, :, 0]  # intensity
    else:
        print("Error: Wrong number of channels: %s" % num_channels)
        exit()
    np_img_transformed = lyft2kitti_conv.transform(np_img)
    # add shift to compensate the shift of UNIT transformation
    #np_img_transformed = shift_image(np_img_transformed, x_shift=-6, y_shift=1)
    #np_img_transformed = shift_image(np_img_transformed, x_shift=1, y_shift=-2)
    np_img_output = np.zeros((width, width, 3))
    if num_channels == 1:
        np_img_output[:, :, 1] = np_img_transformed[0, :, :]  # height
    elif num_channels == 2:
        np_img_output[:, :, 0] = np_img_transformed[0, :, :]  # density
        np_img_output[:, :, 1] = np_img_transformed[1, :, :]  # height
    elif num_channels == 3:
        np_img_output[:, :, 0] = np_img_transformed[0, :, :]  # density
        np_img_output[:, :, 1] = np_img_transformed[1, :, :]  # height
        np_img_output[:, :, 2] = np_img_transformed[2, :, :]  # intensity
    else:
        print("Error: Wrong number of channels: %s" % num_channels)
        exit()
    return np_img_output


def get_target(labels, calib, class_name_to_id_dict):
    labels_bev, noObjectLabels = bev_utils.read_labels_for_bevbox(labels, class_name_to_id_dict)
    if not noObjectLabels:
        labels_bev[:, 1:] = aug_utils.camera_to_lidar_box(labels_bev[:, 1:], calib.V2C, calib.R0,
                                                          calib.P)  # convert rect cam to velo cord
    else:
        print("NO OBJECT LABELS!")
        sys.exit()

    target = bev_utils.build_yolo_target(labels_bev)
    return target


def extract_objects(bev_img, target, draw_bbox=False):
    objects_list = []
    objects_class_list = []

    if draw_bbox:
        bev_utils.draw_box_in_bev(bev_img, target)

    for curr_obj in target:
        # only add object, if target has values inside (target=[0, 0, 0, 0, 0, 0, 0] means that there is no object)
        if not np.array_equal(curr_obj, np.zeros((7,))):
            # get_object_coordinates
            w = curr_obj[3] * cnf.BEV_WIDTH
            l = curr_obj[4] * cnf.BEV_HEIGHT
            crop_size = max(w, l)
            min_x = min(max(0, int(curr_obj[1] * cnf.BEV_WIDTH - crop_size / 2)), cnf.BEV_WIDTH)
            max_x = min(max(0, int(min_x + crop_size)), cnf.BEV_WIDTH)
            min_y = min(max(0, int(curr_obj[2] * cnf.BEV_HEIGHT - crop_size / 2)), cnf.BEV_HEIGHT)
            max_y = min(max(0, int(min_y + crop_size)), cnf.BEV_HEIGHT)

            object_img = bev_img[min_y:max_y, min_x:max_x, :]
            objects_list.append(object_img)
            objects_class_list.append(curr_obj[0])

    return objects_class_list, objects_list


def save_whole_bev_img_with_bboxes(img, target, dataset_name):
    img_output = np.zeros((img.shape[1], img.shape[2], 3))
    img_output[:, :, 0] = img[0, :, :]
    img_output[:, :, 1] = img[1, :, :]
    img_output[:, :, 2] = img[2, :, :]
    bev_utils.draw_box_in_bev(img_output, target)
    imageio.imwrite(dataset_name+".png", img_output)


def save_pixel_values(img, title, show=False):
    gray = cm.get_cmap('gray', 256)
    normalize = colors.Normalize(vmin=0, vmax=255, clip=False)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=gray, norm=normalize)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ax.text(j, i, img[i, j], ha="center", va="center", color="gray", fontsize=3)

    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    plt.savefig(title.replace(' ', '_'), dpi=400)
    plt.cla()
    plt.clf()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_index", type=int, default=0, help="File index of ComplexYOLO validation list")
    parser.add_argument("--dataset", type=str, default="None", help="chose dataset (lyft2kitti2, audi2kitti)")
    parser.add_argument("--num_channels", type=int, default=None, help="Number of channels")
    parser.add_argument('--unit_config', type=str, default=None, help="UNIT net configuration")
    parser.add_argument('--unit_checkpoint', type=str, default=None, help="checkpoint of UNIT autoencoders")
    opt = parser.parse_args()
    print(opt)

    # get specific information to chosen dataset
    dataset, dataset_name, chosen_eval_files_path, get_lidar_label_calib = get_dataset_info(opt.dataset)

    unit_conv = UnitConverter(opt.unit_config, opt.unit_checkpoint)

    # get validation images which are chosen for evaluation
    filename_list = [x.strip() for x in open(chosen_eval_files_path).readlines()]

    filename = filename_list[opt.file_index]
    print("Processing: ", filename)

    lidar, labels, calib = get_lidar_label_calib(dataset, filename)

    b = bev_utils.removePoints(lidar, cnf.boundary)
    bev_array_raw = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary, opt.num_channels)
    # transform to "normal" image array structure
    bev_array = np.zeros((bev_array_raw.shape[1], bev_array_raw.shape[2], 3))
    bev_array[:, :, 0] = bev_array_raw[2, :, :]
    bev_array[:, :, 1] = bev_array_raw[1, :, :]
    bev_array[:, :, 2] = bev_array_raw[0, :, :]
    bev_array_transformed = perform_img2img_translation(unit_conv, bev_array, opt.num_channels)
    #bev_array_transformed = postprocessing.blacken_pixel(bev_array_transformed, threshold=0.1953125)
    #bev_array_transformed[:, :, 0] = postprocessing.density_hist_matching(bev_array_transformed[:, :, 0])

    # convert from float32 to int8
    bev_original_int = (np.round_(bev_array * 255)).astype(np.uint8)
    bev_transformed_int = (np.round_(bev_array_transformed * 255)).astype(np.uint8)

    # extract object images
    target = get_target(labels, calib, dataset.CLASS_NAME_TO_ID)
    objects_class_list, objects_list_original = extract_objects(bev_original_int, target, False)
    objects_class_list, objects_list_transformed = extract_objects(bev_transformed_int, target, False)

    print("%s objects found" % len(objects_list_original))
    for idx in range(len(objects_list_original)):
        print("Processing object %s" % idx)
        object_class_name = list(dataset.CLASS_NAME_TO_ID.keys())[list(dataset.CLASS_NAME_TO_ID.values()).index(int(objects_class_list[idx]))]
        # only height
        if opt.num_channels == 1:
            save_pixel_values(objects_list_original[idx][:, :, 1], title='object %s %s %s height' % (idx, object_class_name, dataset_name))
            save_pixel_values(objects_list_transformed[idx][:, :, 1], title='object %s %s %s2KITTI height' % (idx, object_class_name, dataset_name))
        # density and height
        elif opt.num_channels == 2:
            save_pixel_values(objects_list_original[idx][:, :, 1], title='object %s %s %s height' % (idx, object_class_name, dataset_name))
            save_pixel_values(objects_list_original[idx][:, :, 0], title='object %s %s %s density' % (idx, object_class_name, dataset_name))
            save_pixel_values(objects_list_transformed[idx][:, :, 1], title='object %s %s %s2KITTI height' % (idx, object_class_name, dataset_name))
            save_pixel_values(objects_list_transformed[idx][:, :, 0], title='object %s %s %s2KITTI density' % (idx, object_class_name, dataset_name))
        # density, height and intensity
        elif opt.num_channels == 3:
            save_pixel_values(objects_list_original[idx][:, :, 2], title='object %s %s %s intensity' % (idx, object_class_name, dataset_name))
            save_pixel_values(objects_list_original[idx][:, :, 1], title='object %s %s %s height' % (idx, object_class_name, dataset_name))
            save_pixel_values(objects_list_original[idx][:, :, 0], title='object %s %s %s density' % (idx, object_class_name, dataset_name))
            save_pixel_values(objects_list_transformed[idx][:, :, 2], title='object %s %s %s2KITTI intensity' % (idx, object_class_name, dataset_name))
            save_pixel_values(objects_list_transformed[idx][:, :, 1], title='object %s %s %s2KITTI height' % (idx, object_class_name, dataset_name))
            save_pixel_values(objects_list_transformed[idx][:, :, 0], title='object %s %s %s2KITTI density' % (idx, object_class_name, dataset_name))
        else:
            print("Error: Wrong number of channels: %s" % opt.num_channels)
            exit()
    save_whole_bev_img_with_bboxes(bev_original_int, target, dataset_name)
    save_whole_bev_img_with_bboxes(bev_transformed_int, target, dataset_name+"2KITTI")


if __name__ == '__main__':
    main()
