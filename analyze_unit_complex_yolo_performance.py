import argparse
import re
import matplotlib.pyplot as plt


def read_txt_file(path):
    with open(path) as f:
        return f.readlines()


def search_reg(file_content, regex):
    file_content_str = '\n'.join(file_content)
    p = re.compile(regex)
    return p.findall(file_content_str)


def create_plot(array, title, label, xlabel, ylabel):
    x_labels = [(x+1)*500 for x in range(len(array))]
    plt.plot(x_labels, array, label=label)
    #plt.ylim([0, 1])
    #plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.title(title)


def show_curr_plot():
    plt.show()


def save_curr_plot(output_filename):
    plt.savefig(output_filename, dpi=300)


def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--evaluation_file', type=str, default=None, help="file containing all ComplexYOLO evaluation results")
    #opts = parser.parse_args()
    #evaluation_file = "/home/user/work/master_thesis/code/UNIT/outputs/unit_bev_new_lyft2kitti_2channel_folder_2/unit_bev_new_lyft2kitti_2channel_folder_2.txt"
    evaluation_file = "/home/user/work/master_thesis/code/UNIT/outputs/unit_bev_new_lyft2kitti_2channel_folder/unit_bev_new_lyft2kitti_2channel_folder.txt"

    file_content = read_txt_file(evaluation_file)

    # extract mAPs
    regex = 'mAP:\s(.*)\\n'
    extracted_mAPs = search_reg(file_content, regex)
    extracted_mAPs = list(map(float, extracted_mAPs))
    print(extracted_mAPs)
    print("Max mAP: %s (idx %s)" % (max(extracted_mAPs), extracted_mAPs.index(max(extracted_mAPs))))

    # extract car APs
    regex = "\+\sClass\s'0'\s\(Car\)\s-\sAP:\s(.*)\\n"
    extracted_car_aps = search_reg(file_content, regex)
    extracted_car_aps = list(map(float, extracted_car_aps))
    print("Max car AP: %s (idx %s)" % (max(extracted_car_aps), extracted_car_aps.index(max(extracted_car_aps))))

    # extract pedestrian APs
    regex = "\+\sClass\s'1'\s\(Pedestrian\)\s-\sAP:\s(.*)\\n"
    extracted_pedestrian_aps = search_reg(file_content, regex)
    extracted_pedestrian_aps = list(map(float, extracted_pedestrian_aps))
    print("Max pedestrian AP: %s (idx %s)" % (max(extracted_pedestrian_aps), extracted_pedestrian_aps.index(max(extracted_pedestrian_aps))))

    # extract cyclist APs
    regex = "\+\sClass\s'2'\s\(Cyclist\)\s-\sAP:\s(.*)\\n"
    extracted_cyclist_aps = search_reg(file_content, regex)
    extracted_cyclist_aps = list(map(float, extracted_cyclist_aps))
    print("Max cyclist AP: %s (idx %s)" % (max(extracted_cyclist_aps), extracted_cyclist_aps.index(max(extracted_cyclist_aps))))

    # create plot
    plot_title = "ComplexYOLO mAP"
    axis_label_x = "iterations"
    axis_label_y = "ComplexYOLO AP"
    create_plot(extracted_mAPs, plot_title, "mAP", axis_label_x, axis_label_y)
    create_plot(extracted_car_aps, plot_title, "car AP", axis_label_x, axis_label_y)
    create_plot(extracted_pedestrian_aps, plot_title, "pedestrian AP", axis_label_x, axis_label_y)
    create_plot(extracted_cyclist_aps, plot_title, "cyclist AP", axis_label_x, axis_label_y)
    show_curr_plot()
    #save_curr_plot("complexYOLO_eval.png")


if __name__ == '__main__':
    main()
