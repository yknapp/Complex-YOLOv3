import os
import glob
import numpy as np


def read_calib_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R_rect': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


#calib_path = "/home/user/work/master_thesis/datasets/kitti/kitti/object/training/calib/*.txt"
calib_path = "/home/user/work/master_thesis/datasets/lyft_kitti/object/training/calib/*.txt"
calib_filenames = glob.glob(calib_path)

Tr_velo_to_cam_addition = np.array([[.0, .0, .0, .0], [.0, .0, .0, .0], [.0, .0, .0, .0]])
R0_addition = np.array([[.0, .0, .0], [.0, .0, .0], [.0, .0, .0]])
P2_addition = np.array([[.0, .0, .0, .0], [.0, .0, .0, .0], [.0, .0, .0, .0]])

for calib_filename in calib_filenames:
    calib_file_path = os.path.join(calib_path, calib_filename)
    calib = read_calib_file(calib_file_path)
    Tr_velo_to_cam_addition += calib['Tr_velo2cam']
    R0_addition += calib['R_rect']
    P2_addition += calib['P2']

print("Tr_velo_to_cam Mean: ", np.true_divide(Tr_velo_to_cam_addition, len(calib_filenames)))
print("R0 Mean: ", np.true_divide(R0_addition, len(calib_filenames)))
print("P2 Mean: ", np.true_divide(P2_addition, len(calib_filenames)))
