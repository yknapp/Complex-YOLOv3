import numpy as np


class Config:
	def __init__(self):
		self.root_dir = None  # path to dataset
		self.class_list = ["Car", "Pedestrian", "Cyclist"]
		self.colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]  # bounding box RGB colors for classes
		self.CLASS_NAME_TO_ID = dict()  # mapping of class names to class IDs
		self.boundary = dict()
		self.BEV_WIDTH = None  # width of BEV image
		self.BEV_HEIGHT = None  # height of BEV image
		self.DISCRETIZATION = None  # discretization of 3D pointcloud to 2D BEV image
		self.Tr_velo_to_cam = None
		self.R0 = None
		self.P2 = None
		self.R0_inv = None
		self.Tr_velo_to_cam_inv = None
		self.P2_inv = None

	def calc_discretization(self):
		return (self.boundary["maxX"] - self.boundary["minX"])/self.BEV_HEIGHT


class KittiConfig(Config):
	def __init__(self):
		super().__init__()
		self.root_dir = '/home/user/work/master_thesis/datasets/kitti/kitti'
		self.BEV_WIDTH = 480
		self.BEV_HEIGHT = 480
		self.DISCRETIZATION = self.calc_discretization()
		self.CLASS_NAME_TO_ID = {
			'Car': 0,
			'Pedestrian': 1,
			'Cyclist': 2,
			'Van': 0,
			'Person_sitting': 1,
		}

		# Following parameters are calculated as an average from KITTI dataset for simplicity
		self.Tr_velo_to_cam = np.array([
			[7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
			[1.18652889e-02,  9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
			[9.99882833e-01,  7.49141178e-03,  1.18719929e-02, -2.78557062e-01],
			[			  0,               0,               0,               1]
		])

		# cal mean from train set
		self.R0 = np.array([
				[0.99992475, 0.00975976, -0.00734152, 0],
				[-0.0097913, 0.99994262, -0.00430371, 0],
				[0.00729911,  0.0043753,  0.99996319, 0],
				[		  0,          0,           0, 1]
		])

		self.P2 = np.array([
				[719.787081, 		 0., 608.463003,    44.9538775],
				[		 0., 719.787081, 174.545111,     0.1066855],
				[		 0.,         0.,         1., 3.0106472e-03],
				[		 0.,         0.,         0.,             0]
		])

		self.R0_inv = np.linalg.inv(self.R0)
		self.Tr_velo_to_cam_inv = np.linalg.inv(self.Tr_velo_to_cam)
		self.P2_inv = np.linalg.pinv(self.P2)


class LyftConfig(Config):
	def __init__(self):
		super().__init__()
		self.root_dir = '/home/user/work/master_thesis/datasets/lyft_kitti'
		self.BEV_WIDTH = 480
		self.BEV_HEIGHT = 480
		self.DISCRETIZATION = self.calc_discretization()
		self.CLASS_NAME_TO_ID = {
			'Car': 0,
			'Pedestrian': 1,
			'Cyclist': 2,
			'Van': 0,
			'Person_sitting': 1
		}

		# Following parameters are calculated as an average from KITTI dataset for simplicity
		self.Tr_velo_to_cam = np.array([
			[ 0.06126085, -0.99807662, -0.00551671, -0.03957647],
			[-0.00818966,   0.0050224,   -0.999943, -0.14756945],
			[ 0.99806068,  0.06130232, -0.00784974, -0.31270448],
			[		  0., 		   0., 			0.,			 1.]
		])

		# cal mean from train set
		self.R0 = np.array([
			[1.0, 0.0, 0.0, 0.0],
			[0.0, 1.0, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0]
		])

		self.P2 = np.array([
			[881.68854969, 			 0., 624.97773644, 0.],
			[		   0., 881.68854969, 522.05127608, 0.],
			[		   0., 			 0., 		   1., 0.],
			[		   0., 			 0.,		   0., 0.]
		])

		self.R0_inv = np.linalg.inv(self.R0)
		self.Tr_velo_to_cam_inv = np.linalg.inv(self.Tr_velo_to_cam)
		self.P2_inv = np.linalg.pinv(self.P2)
