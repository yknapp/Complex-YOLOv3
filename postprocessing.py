import numpy as np
import cv2
import random
import utils.config as cnf
import utils.dataset_bev_utils as bev_utils
from utils.kitti_yolo_dataset import KittiYOLODataset


def read_txt_file(path):
    print("Reading text file '%s'" % path)
    with open(path) as f:
        return f.readlines()


def blacken_pixel(np_img, threshold):
    return np.where(np_img > threshold, np_img, 0)


def calculate_cdf(histogram):
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def histogram_matching(np_img_src, np_img_ref):
    # compute histograms of source and reference image
    src_hist, bin_0 = np.histogram(np_img_src.flatten(), 256, [0, 256])
    ref_hist, bin_1 = np.histogram(np_img_ref.flatten(), 256, [0, 256])

    # compute normalized cdf for the source and reference image
    src_cdf = calculate_cdf(src_hist)
    ref_cdf = calculate_cdf(ref_hist)

    # create lookup table from source to reference image
    lookup_table = calculate_lookup(src_cdf, ref_cdf)

    # use lookup function to transform the source image
    src_after_transform = cv2.LUT(np_img_src, lookup_table)

    # performs scaling, taking an absolute value, conversion to an unsigned 8-bit type on each element of the input array
    src_after_transform = cv2.convertScaleAbs(src_after_transform)

    return src_after_transform


def density_hist_matching(density_channel):
    # map given density channel to [0, 255], so histogram matching can be applied
    density_channel_int = (np.round_(density_channel * 255)).astype(np.uint8)

    # get kitti BEV image
    dataset = KittiYOLODataset(split='valid', mode='EVAL', folder='training', data_aug=False)
    kitti_valid_textfile = read_txt_file("/home/user/work/master_thesis/code/Complex-YOLOv3/data/KITTI/ImageSets/valid.txt")
    kitti_valid_filename_list = [x.strip() for x in kitti_valid_textfile]
    random_idx = random.randint(0, len(kitti_valid_filename_list)-1)
    random_filename = kitti_valid_filename_list[random_idx]
    print("Take random KITTI file '%s'" % random_filename)
    kitti_lidar = dataset.get_lidar(int(random_filename))
    b = bev_utils.removePoints(kitti_lidar, cnf.boundary)
    # b = bev_utils.remove_fov_points(b, calib)  # remove points outside camera FOV
    kitti_bev = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
    density_channel_kitti = kitti_bev[2, :, :]
    density_channel_kitti_int = (np.round_(density_channel_kitti * 255)).astype(np.uint8)

    # convert from float32 to int8
    density_channel_int_matched = histogram_matching(density_channel_int, density_channel_kitti_int)
    # convert back from int8 to float32
    density_channel_matched = np.true_divide(density_channel_int_matched, 255).astype(np.float32)

    return density_channel_matched



def create_histogram(image, num_bin):
    return cv2.calcHist([image], [0], None, [num_bin], [0, 255])


def plot_histogram(histogram, label, title, show=False, clear=False):
    import matplotlib.pyplot as plt
    if clear:
        # clear plot from previous data
        plt.clf()
    plt.plot(histogram, label=label)
    plt.xlim(right=histogram.shape[0])
    plt.yscale('log')
    plt.xlabel('Grayscale value')
    plt.ylabel('Number of pixels')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


def plot_image(img):
    import matplotlib.pyplot as plt
    plt.imshow(img, interpolation='nearest')
    plt.show()


def main():
    # blacken image < 50 test
    #a = np.arange(100)
    #print(a)
    #print(blacken_pixel(a, 50))

    # test histogram matching
    # get lyft2kitti BEV image
    from utils.lyft2kitti_yolo_dataset2 import Lyft2KittiYOLODataset2
    dataset = Lyft2KittiYOLODataset2(split='valid', mode='EVAL', folder='training', data_aug=False)
    lyft_bev = dataset.get_bev("c55e6837b0a561e929308ec21b7be251250eb817c62441b82f3becdfede7c1f7")
    density_channel = lyft_bev[2, :, :]
    density_channel_int = (np.round_(density_channel * 255)).astype(np.uint8)
    hist_density = create_histogram(density_channel_int, 255)
    plot_histogram(hist_density, label='Lyft', title='BEV Histogram Density')

    # get kitti BEV image
    import utils.config as cnf
    import utils.dataset_bev_utils as bev_utils
    from utils.kitti_yolo_dataset import KittiYOLODataset
    dataset = KittiYOLODataset(split='valid', mode='EVAL', folder='training', data_aug=False)
    sample_id = 200
    kitti_lidar = dataset.get_lidar(sample_id)
    b = bev_utils.removePoints(kitti_lidar, cnf.boundary)
    # b = bev_utils.remove_fov_points(b, calib)  # remove points outside camera FOV
    kitti_bev = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
    density_channel_kitti = kitti_bev[2, :, :]
    density_channel_kitti_int = (np.round_(density_channel_kitti * 255)).astype(np.uint8)
    hist_density_kitti = create_histogram(density_channel_kitti_int, 255)
    plot_histogram(hist_density_kitti, label='KITTI', title='BEV Histogram Density')

    # convert from float32 to int8
    density_channel_int_matched = histogram_matching(density_channel_int, density_channel_kitti_int)
    # convert back from int8 to float32
    density_channel_matched = np.true_divide(density_channel_int_matched, 255).astype(np.float32)

    hist_density_matched = create_histogram(density_channel_int_matched, 255)
    plot_histogram(hist_density_matched, label='Lyft2KITTI_matched', title='BEV Histogram Density')

    #plot_image(density_channel_int)
    #plot_image(density_channel_int_matched)


if __name__ == '__main__':
    main()
