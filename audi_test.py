import json
import pprint
import os
import numpy as np
import numpy.linalg as la
import utils.dataset_bev_utils as bev_utils
import utils.dataset_aug_utils as aug_utils


AUDI_ROOT_DIR = "/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes"
# lidar boundarys for Bird's Eye View
BOUNDARY = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}
IMG_HEIGHT = 480
IMG_WIDTH = 480

AUDI_CLASS_NAME_TO_ID = {
            'Car': 				    0,
            'Pedestrian': 		    1,
            'Bicycle': 			    2
            }


class Object3dLabel(object):
    def __init__(self, bbox):
        self.class_name = bbox['class']
        self.truncation = float(bbox['truncation'])
        self.occlusion = float(bbox['occlusion'])
        self.alpha = float(bbox['alpha'])

        # 2d bounding box
        self.xmin = float(bbox['2d_bbox'][1])  # left
        self.xmax = float(bbox['2d_bbox'][3])  # right
        self.ymin = float(bbox['2d_bbox'][0])  # top
        self.ymax = float(bbox['2d_bbox'][2])  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # 3d bounding box
        self.h = float(bbox['size'][2])  # bbox height
        self.w = float(bbox['size'][0])  # bbox width
        self.l = float(bbox['size'][1])  # bbox length (in meters)
        center_x = float(bbox['center'][0])
        center_y = float(bbox['center'][1])
        center_z = float(bbox['center'][2])
        self.t = (center_x, center_y, center_z)  # location (x,y,z) in camera coord.
        self.ry = float(bbox['rot_angle'])
        self.score = -1.0


class Calibration:
    def __init__(self, sensor_config_filepath):
        calibration_config = open_config(sensor_config_filepath)
        lidars_front_center_view = calibration_config['lidars']['rear_left']['view']
        transform_to_global = self.get_transform_to_global(lidars_front_center_view)
        self.V2C = transform_to_global[:3, :]
        self.R0 = self.get_rot_from_global(lidars_front_center_view)
        cameras_front_center_view = calibration_config['cameras']['front_center']['view']
        transform_global_to_camera = self.get_transform_from_global(cameras_front_center_view)
        self.P = transform_global_to_camera[:3, :]

    def get_axes_of_a_view(self, view):
        EPSILON = 1.0e-10  # norm should not be small
        x_axis = view['x-axis']
        y_axis = view['y-axis']

        x_axis_norm = la.norm(x_axis)
        y_axis_norm = la.norm(y_axis)

        if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
            raise ValueError("Norm of input vector(s) too small.")

        # normalize the axes
        x_axis = x_axis / x_axis_norm
        y_axis = y_axis / y_axis_norm

        # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
        y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

        # create orthogonal z-axis
        z_axis = np.cross(x_axis, y_axis)

        # calculate and check y-axis and z-axis norms
        y_axis_norm = la.norm(y_axis)
        z_axis_norm = la.norm(z_axis)

        if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
            raise ValueError("Norm of view axis vector(s) too small.")

        # make x/y/z-axes orthonormal
        y_axis = y_axis / y_axis_norm
        z_axis = z_axis / z_axis_norm

        return x_axis, y_axis, z_axis

    def get_origin_of_a_view(self, view):
        return view['origin']

    def get_transform_to_global(self, view):
        # get axes
        x_axis, y_axis, z_axis = self.get_axes_of_a_view(view)

        # get origin
        origin = self.get_origin_of_a_view(view)
        transform_to_global = np.eye(4)

        # rotation
        transform_to_global[0:3, 0] = x_axis
        transform_to_global[0:3, 1] = y_axis
        transform_to_global[0:3, 2] = z_axis

        # origin
        transform_to_global[0:3, 3] = origin

        return transform_to_global

    def get_rot_from_global(self, view):
        # get transform to global
        transform_to_global = self.get_transform_to_global(view)
        # get rotation
        rot = np.transpose(transform_to_global[0:3, 0:3])
        return rot

    def transform_from_to(self, src, target):
        transform = np.dot(self.get_transform_from_global(target), self.get_transform_to_global(src))
        return transform

    def get_transform_from_global(self, view):
        # get transform to global
        transform_to_global = self.get_transform_to_global(view)
        trans = np.eye(4)
        rot = np.transpose(transform_to_global[0:3, 0:3])
        trans[0:3, 0:3] = rot
        trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])

        return trans


def map_lidar_to_label_filename(file_name):
    label_filename_template = '%s_label3D_frontcenter_%s.json'
    file_name_list = file_name.replace('.npz', '').split('_')
    timestamp = file_name_list[0]
    idx = file_name_list[3]
    label_filename = label_filename_template % (timestamp, idx)
    return label_filename


def open_config(config_filepath):
    with open(config_filepath, 'r') as f:
        config = json.load(f)
    return config


def load_lidar_file(file_path):
    lidar_pc_raw = np.load(file_path)
    lidar_pc = np.zeros([lidar_pc_raw['points'].shape[0], 4])
    lidar_pc[:, :3] = lidar_pc_raw['points']
    lidar_pc[:, 3] = lidar_pc_raw['reflectance']
    return lidar_pc


def get_label(file_path):
    bboxs = open_config(file_path)
    pprint.pprint(bboxs)
    objects = [Object3dLabel(bboxs[bbox]) for bbox in bboxs.keys()]
    return objects


def get_calibration():
    sensor_config_filepath = '/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/cams_lidars.json'
    calib = Calibration(sensor_config_filepath)
    return calib


def save_np_as_img(np_array, output_filename="labeled_bev"):
    from PIL import Image
    im = Image.fromarray(np_array)
    im.save("%s.png" % output_filename)
    print("Saved labeled BEV image to '%s.png'" % output_filename)


def load_calibration_file():
    # load calibration file
    sensor_config_filepath = '/home/user/work/master_thesis/datasets/audi/camera_lidar_semantic_bboxes/cams_lidars.json'
    config = open_config(sensor_config_filepath)
    pprint.pprint(config)


def main():
    lidar_filename = "20181108123750_lidar_frontcenter_000045104.npz"
    label_filename = map_lidar_to_label_filename(lidar_filename)
    lidar_path = os.path.join(AUDI_ROOT_DIR, 'lidar', 'cam_front_center', lidar_filename)
    label_path = os.path.join(AUDI_ROOT_DIR, 'label3D', 'cam_front_center', label_filename)
    lidar_pc = load_lidar_file(lidar_path)
    # filter point cloud points inside fov
    lidar_pc_filtered = bev_utils.removePoints(lidar_pc, BOUNDARY)
    # create Bird's Eye View
    discretization = (BOUNDARY["maxX"] - BOUNDARY["minX"]) / IMG_HEIGHT
    lidar_bev = bev_utils.makeBVFeature(lidar_pc_filtered, discretization, BOUNDARY)

    lidar_bev = (np.round_(lidar_bev * 255)).astype(np.uint8)
    channel, height, width = lidar_bev.shape
    lidar_bev_2 = np.zeros((width, width, 3)).astype(np.uint8)
    lidar_bev_2[:, :, 0] = lidar_bev[0, :, :]
    lidar_bev_2[:, :, 1] = lidar_bev[1, :, :]

    labels = get_label(label_path)
    calib = get_calibration()
    labels_bev, noObjectLabels = bev_utils.read_labels_for_bevbox(labels, AUDI_CLASS_NAME_TO_ID)
    if not noObjectLabels:
        labels_bev[:, 1:] = aug_utils.camera_to_lidar_box(labels_bev[:, 1:], calib.V2C, calib.R0, calib.P)  # convert rect cam to velo cord

    target = bev_utils.build_yolo_target(labels_bev)
    bev_utils.draw_box_in_bev(lidar_bev_2, target)

    save_np_as_img(lidar_bev_2, output_filename="labeled_bev")

    #load_calibration_file()


if __name__ == '__main__':
    main()
