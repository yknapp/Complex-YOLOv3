#!/bin/bash

dataset=kitti
num_channel=3

python analyze_mean_histogram.py --dataset $dataset --num_channel $num_channel
