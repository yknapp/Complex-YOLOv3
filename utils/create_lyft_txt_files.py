import glob

lyft_lidar_dir = "/home/user/datasets/lyft_kitti/object/training/velodyne/*.bin"
lidar_filenames = glob.glob(lyft_lidar_dir)
number = 1481

text_file = open("Output.txt", "w")
for idx in range(number):
    file_path = lidar_filenames[idx]
    filename = file_path.split('/')[-1].replace('.bin', '')
    text_file.write(filename + '\n')
text_file.close()
