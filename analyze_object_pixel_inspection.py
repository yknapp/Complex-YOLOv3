import sys
import argparse
import numpy as np
import utils.config as cnf
import utils.dataset_bev_utils as bev_utils
import utils.dataset_aug_utils as aug_utils
import matplotlib.pyplot as plt

from unit.unit_converter import UnitConverter


def get_dataset_info(dataset):
    if dataset == 'lyft2kitti2':
        dataset_name = 'Lyft'
        chosen_eval_files_path = 'data/LYFT/ImageSets/valid.txt'
        bev_output_path = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/bev'
        from utils.lyft2kitti_dataset2 import Lyft2KittiDataset
        dataset = Lyft2KittiDataset()
    else:
        print("Unknown dataset '%s'" % dataset)
        sys.exit()
    return dataset, dataset_name, chosen_eval_files_path, bev_output_path


def perform_img2img_translation(lyft2kitti_conv, np_img):
    height, width, c = np_img.shape
    np_img_transformed = lyft2kitti_conv.transform(np_img)
    np_img_output = np.zeros((width, width, 2))
    np_img_output[:, :, 0] = np_img_transformed[0, :, :]
    np_img_output[:, :, 1] = np_img_transformed[1, :, :]
    return np_img_output


def extract_objects(bev_img, labels, calib, class_name_to_id_dict, draw_bbox=False):
    objects_list = []
    objects_class_list = []
    labels_bev, noObjectLabels = bev_utils.read_labels_for_bevbox(labels, class_name_to_id_dict)
    if not noObjectLabels:
        labels_bev[:, 1:] = aug_utils.camera_to_lidar_box(labels_bev[:, 1:], calib.V2C, calib.R0,
                                                          calib.P)  # convert rect cam to velo cord
    else:
        print("NO OBJECT LABELS!")
        sys.exit()

    target = bev_utils.build_yolo_target(labels_bev)
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


def save_pixel_values(img, title, show=False):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ax.text(j, i, img[i, j], ha="center", va="center", color="gray", fontsize=3)

    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    plt.savefig(title.replace(' ', '_'), dpi=400)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_index", type=int, default=0, help="File index of ComplexYOLO validation list")
    parser.add_argument("--dataset", type=str, default="None", help="chose dataset (lyft2kitti2, audi2kitti)")
    parser.add_argument("--model_def", type=str, default="config/complex_tiny_yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/tiny-yolov3_ckpt_epoch-220.pth",
                        help="path to weights file")
    parser.add_argument('--unit_config', type=str, default=None, help="UNIT net configuration")
    parser.add_argument('--unit_checkpoint', type=str, default=None, help="checkpoint of UNIT autoencoders")
    opt = parser.parse_args()

    # get specific information to chosen dataset
    dataset, dataset_name, chosen_eval_files_path, bev_output_path = get_dataset_info(opt.dataset)

    unit_conv = UnitConverter(opt.unit_config, opt.unit_checkpoint)

    # get validation images which are chosen for evaluation
    image_filename_list = [x.strip() for x in open(chosen_eval_files_path).readlines()]

    image_filename = image_filename_list[opt.file_index]
    print("Processing: ", image_filename)
    lidar = dataset.get_lidar(image_filename)
    labels = dataset.get_label(image_filename)
    calib = dataset.get_calib(image_filename)

    b = bev_utils.removePoints(lidar, cnf.boundary)
    bev_array_raw = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
    bev_array = np.zeros((bev_array_raw.shape[1], bev_array_raw.shape[2], 2))
    bev_array[:, :, 0] = bev_array_raw[2, :, :]
    bev_array[:, :, 1] = bev_array_raw[1, :, :]
    bev_array_transformed = perform_img2img_translation(unit_conv, bev_array)

    # convert from float32 to int8
    bev_original_int = (np.round_(bev_array * 255)).astype(np.uint8)
    bev_transformed_int = (np.round_(bev_array_transformed * 255)).astype(np.uint8)

    # extract objects
    objects_class_list, objects_list_original = extract_objects(bev_original_int, labels, calib, dataset.CLASS_NAME_TO_ID, False)
    objects_class_list, objects_list_transformed = extract_objects(bev_transformed_int, labels, calib, dataset.CLASS_NAME_TO_ID, False)

    print("%s objects found" % len(objects_list_original))
    for idx in range(len(objects_list_original)):
        print("Processing object %s" % idx)
        object_class_name = list(dataset.CLASS_NAME_TO_ID.keys())[list(dataset.CLASS_NAME_TO_ID.values()).index(int(objects_class_list[idx]))]
        save_pixel_values(objects_list_original[idx][:, :, 1], title='object %s %s %s' % (idx, object_class_name, dataset_name))
        save_pixel_values(objects_list_transformed[idx][:, :, 1], title='object %s %s %s2KITTI' % (idx, object_class_name, dataset_name))


if __name__ == '__main__':
    main()
