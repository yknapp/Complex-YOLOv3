import torch
import numpy as np

root_dir = '/home/user/work/master_thesis/datasets/lyft_kitti'

class_list = ["Car", "Pedestrian", "Cyclist"]

CLASS_NAME_TO_ID = {
	'Car': 				0,
	'Pedestrian': 		1,
	'Cyclist': 			2,
	'Van': 				0,
	'Person_sitting': 	1,
}

# Front side (of vehicle) Point Cloud boundary for BEV
boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

# Back back (of vehicle) Point Cloud boundary for BEV
boundary_back = {
    "minX": -50,
    "maxX": 0,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

#BEV_WIDTH = 608 # across y axis -25m ~ 25m
BEV_WIDTH = 480  # across y axis -25m ~ 25m
#BEV_HEIGHT = 608 # across x axis 0m ~ 50m
BEV_HEIGHT = 480  # across x axis 0m ~ 50m

DISCRETIZATION = (boundary["maxX"] - boundary["minX"])/BEV_HEIGHT

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]

# Following parameters are calculated as an average from Lyft dataset for simplicity
#####################################################################################
Tr_velo_to_cam = np.array([
		[0.06126085, -0.99807662, -0.00551671, -0.03957647],
		[-0.00818966, 0.0050224, -0.999943, -0.14756945],
		[0.99806068, 0.06130232, -0.00784974, -0.31270448],
		[0, 0, 0, 1]
	])

# cal mean from train set
R0 = np.array([
		[1.0, .0, .0, .0],
		[.0, 1.0, .0, .0],
		[.0, .0, 1.0, .0],
		[.0, .0, .0, 1.0]
])

P2 = np.array([[881.68854969,         0., 624.97773644,    0.],
               [        0., 881.68854969, 522.05127608,    0.],
               [        0.,         0.,         1., 0.],
			   [0., 0., 0., 0]
])

R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)
#####################################################################################