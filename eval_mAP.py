from __future__ import division

from models import *
from utils.utils import *

import os, sys, time, datetime, argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

#import utils.config as cnf
import utils.config as cnf

def evaluate(dataset_name, model, iou_thres, conf_thres, nms_thres, img_size, batch_size, num_channels, unit_config_path, unit_checkpoint_path, alter_bev_path=None):
    model.eval()

    # Get dataloader
    split='valid'

    # prepare dataset
    if dataset_name == 'kitti':
        from utils.kitti_yolo_dataset import KittiYOLODataset
        dataset = KittiYOLODataset(split=split, mode='EVAL', num_channels=num_channels, folder='training', data_aug=False)
    elif dataset_name == 'lyft':
        from utils.lyft_yolo_dataset import LyftYOLODataset
        dataset = LyftYOLODataset(split=split, mode='EVAL', num_channels=num_channels, folder='training', data_aug=False)
    elif dataset_name == 'lyft2kitti':
        from utils.lyft2kitti_yolo_dataset import Lyft2KittiYOLODataset
        if None not in (unit_config_path, unit_checkpoint_path):
            dataset = Lyft2KittiYOLODataset(unit_config_path=unit_config_path, unit_checkpoint_path=unit_checkpoint_path, split=split, mode='EVAL', num_channels=num_channels, folder='training', data_aug=False)
        else:
            print("Program arguments 'unit_config' and 'unit_checkpoint' must be set for dataset Lyft2Kitti")
            sys.exit()
    elif dataset_name == 'lyft2kitti2':
        from utils.lyft2kitti_yolo_dataset2 import Lyft2KittiYOLODataset2
        dataset = Lyft2KittiYOLODataset2(split=split, mode='EVAL', num_channels=num_channels, folder='training', data_aug=False, alter_bev_path=alter_bev_path)
    elif dataset_name == 'audi':
        from utils.audi_yolo_dataset import AudiYOLODataset
        dataset = AudiYOLODataset(split=split, mode='EVAL', num_channels=num_channels, data_aug=False)
    elif dataset_name == 'audi2kitti':
        from utils.audi2kitti_yolo_dataset import Audi2KittiYOLODataset
        dataset = Audi2KittiYOLODataset(split=split, mode='EVAL', num_channels=num_channels, data_aug=False, alter_bev_path=alter_bev_path)
    else:
        print("Error: Unknown dataset '%s'" % dataset_name)
        sys.exit()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression_rotated_bbox(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="kitti", help="chose dataset (kitti, lyft, lyft2kitti, lyft2kitti2, audi, audi2kitti)")
    parser.add_argument("--num_channels", type=int, default=None, help="Number of channels")
    parser.add_argument("--batch_size", type=int, default=10, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/complex_tiny_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/tiny-yolov3_ckpt_epoch-220.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument('--unit_config', type=str, help="UNIT net configuration")
    parser.add_argument('--unit_checkpoint', type=str, help="checkpoint of UNIT autoencoders")
    parser.add_argument('--alter_bev_path', type=str, default=None, help="optional alternative path for generated BEVs")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_classes(opt.class_path)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")
    precision, recall, AP, f1, ap_class = evaluate(
        opt.dataset,
        model,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        num_channels=opt.num_channels,
        unit_config_path=opt.unit_config,
        unit_checkpoint_path=opt.unit_checkpoint,
        alter_bev_path=opt.alter_bev_path
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]} - recall: {recall[i]}")

    print(f"mAP: {AP.mean()}")

if __name__ == "__main__":
    main()
