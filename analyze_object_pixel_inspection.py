import os
import numpy as np
import utils.dataset_utils as dataset_utils
import utils.dataset_bev_utils as bev_utils
import utils.dataset_aug_utils as aug_utils
import cv2
import matplotlib.pyplot as plt

BEV_DATASET_PATH = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/bev_arrays'
LABEL_PATH = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/label_2'
CALIB_PATH = '/home/user/work/master_thesis/datasets/lyft_kitti/object/training/calib'
LYFT_CLASS_NAME_TO_ID = {
            'car': 				    0,
            'pedestrian': 		    1,
            'bicycle': 			    2
            }
BEV_WIDTH = 480
BEV_HEIGHT = 480


def get_label(idx):
    label_file = os.path.join(LABEL_PATH, '%s.txt' % idx)
    assert os.path.exists(label_file)
    return dataset_utils.read_label(label_file)


def get_calib(idx):
    calib_file = os.path.join(CALIB_PATH, '%s.txt' % idx)
    assert os.path.exists(calib_file)
    return dataset_utils.Calibration(calib_file)


def load_bevs(filename):
    input_file_path = os.path.join(BEV_DATASET_PATH, filename)
    original_filename = input_file_path+'_original.npy'
    transformed_filename = input_file_path+'_transformed.npy'
    bev_original = np.load(original_filename)
    bev_transformed = np.load(transformed_filename)
    return bev_original, bev_transformed


def extract_objects(bev_img, labels, calib, draw_bbox=False):
    objects_list = []
    labels_bev, noObjectLabels = bev_utils.read_labels_for_bevbox(labels, LYFT_CLASS_NAME_TO_ID)
    if not noObjectLabels:
        labels_bev[:, 1:] = aug_utils.camera_to_lidar_box(labels_bev[:, 1:], calib.V2C, calib.R0,
                                                          calib.P)  # convert rect cam to velo cord

    target = bev_utils.build_yolo_target(labels_bev)
    if draw_bbox:
        bev_utils.draw_box_in_bev(bev_img, target)

    for curr_obj in target:
        # get_object_coordinates
        w = curr_obj[3] * BEV_WIDTH
        l = curr_obj[4] * BEV_HEIGHT
        crop_size = max(w, l)
        min_x = min(max(0, int(curr_obj[1] * BEV_WIDTH - crop_size / 2)), BEV_WIDTH)
        max_x = min(max(0, int(min_x + crop_size)), BEV_WIDTH)
        min_y = min(max(0, int(curr_obj[2] * BEV_HEIGHT - crop_size / 2)), BEV_HEIGHT)
        max_y = min(max(0, int(min_y + crop_size)), BEV_HEIGHT)

        object_img = bev_img[min_y:max_y, min_x:max_x, :]
        objects_list.append(object_img)

    return objects_list


def save_pixel_values(img, title, show=False):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')

    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            ax.text(j, i, img[i, j], ha="center", va="center", color="gray", fontsize=3)

    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    plt.savefig(title.replace(' ', '_'), dpi=400)
    plt.close()

def main():
    filename = "f2219d4920c2505c9c6620877b3ae6ac11fda31c0b6d00cc8248752a862c00d3"
    print("processing ", filename)
    bev_original, bev_transformed = load_bevs(filename)
    labels = get_label(filename)
    calib = get_calib(filename)

    # convert from float32 to int8
    bev_original_int = (np.round_(bev_original * 255)).astype(np.uint8)
    bev_transformed_int = (np.round_(bev_transformed * 255)).astype(np.uint8)

    objects_list_original = extract_objects(bev_original_int, labels, calib, False)
    objects_list_transformed = extract_objects(bev_transformed_int, labels, calib, False)

    for idx in range(len(objects_list_original)):
        print("Processing object %s" % idx)
        save_pixel_values(objects_list_original[idx][:, :, 1], title='%s object %s' % ("KITTI", idx))
        save_pixel_values(objects_list_transformed[idx][:, :, 1], title='%s object %s' % ("Lyft2KITTI", idx))


if __name__ == '__main__':
    main()
