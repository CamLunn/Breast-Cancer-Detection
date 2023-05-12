# the python script to train the masses with 'BENIGN & MALIGNANT' classess. Here we are treating the
# soft_tissue, skin_thickening, and nipple_retraction as a same category - soft_tissue.

# importing the required libraries

# Check Pytorch installation
import torch, torchvision

print(torch.__version__, torch.cuda.is_available())

# importing model's modules
from mmdet.models import build_detector
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.apis import train_detector

# importing packiage to be later use
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
from pathlib import Path
import json
from glob import glob

# Check MMDetection installation
import mmdet

print(mmdet.__version__)
import os

# Check mmcv installation
import mmcv
from mmcv import Config
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

print(get_compiling_cuda_version())
print(get_compiler_version())

# Lets define the paths
root_path = Path('/home/adil/Desktop/PhD_Data/PhD_Courses/First_Semester/Machine_Learning/Breast_Cancer_Detection/CBIS_DDSM/')
train_json =  str (root_path / 'coco_files/annotations/train.json')
test_json =  str(root_path / 'coco_files/annotations/validation.json')
train_images = str(root_path / 'coco_files/train/')
test_images =  str(root_path / 'coco_files/validation')



# importing the config file
config_path = str(root_path / 'Breast_detection_model/model/mask_rcnn/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py')
cfg = Config.fromfile(config_path)


# classes = ('calcification_malignant', 'calcification_benign')
classes = (
    "MASS_BENIGN",
    "MASS_MALIGNANT",
    'CALCIFICATION_BENIGN',
    'CALCIFICATION_MALIGNANT'
)


# Reading dataset type
cfg.dataset_type = "COCODataset"

# reading the data
cfg.data.test.ann_file = test_json
cfg.data.test.img_prefix = test_images
cfg.data.test.classes = classes


cfg.data.train.ann_file = train_json
cfg.data.train.img_prefix = train_images
cfg.data.train.classes = classes


cfg.data.val.ann_file = test_json
cfg.data.val.img_prefix = test_images
cfg.data.val.classes = classes


## Customizing the configuration

# modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = len(classes)
cfg.model.roi_head.mask_head.num_classes = len(classes)

# setting number of samples per_gpu, number of workers and number of epochs
# cfg.data.samples_per_gpu = 1
# cfg.data.workers_per_gpu = 1
cfg.runner.max_epochs = 1

# Uploading the pretrained weights - here we will initialize network with CatchAll Algo weights to obtain a higher performance
# cfg.load_from = './weights/combined_data_weights/epoch_9.pth'

checkpoints_path = str(root_path / 'Breast_detection_model/weights/coco_weights/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth')
cfg.load_from  = checkpoints_path


# Set up working dir to save files and logs.
cfg.work_dir = str(root_path / 'Checkpoints')
# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 1

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1

# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f"Config:\n{cfg.pretty_text}")

# cfg.img_norm_cfg["mean"] = [15.96800848, 15.34300115, 16.94953167]
# cfg.train_pipeline[4]["mean"] = [15.96800848, 15.34300115, 16.94953167]
# cfg.test_pipeline[1]["transforms"][2]["mean"] = [15.96800848, 15.34300115, 16.94953167]
# cfg.data["train"]["pipeline"][4]["mean"] = [15.96800848, 15.34300115, 16.94953167]
# cfg.data["val"]["pipeline"][1]["transforms"][2]["mean"] = [
#     15.96800848,
#     15.34300115,
#     16.94953167,
# ]
# cfg.data.test.pipeline[1].transforms[2].mean = [15.96800848, 15.34300115, 16.94953167]


# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)
model.CLASSES = classes

# Create work_dir and calling the train function
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
