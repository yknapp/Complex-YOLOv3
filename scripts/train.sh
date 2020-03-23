#!/bin/bash

conda activate ComplexYOLO_1.1
python train.py --model_def config/complex_yolov3_low_res.cfg
