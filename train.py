# Import PyTorch and TorchVision
import torch, torchvision

# Import some common libraries
import numpy as np
import os, json, cv2, random

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import (
            COCOEvaluator, DatasetEvaluators
            )
import lsknet

OUTPUT_DIRECTORY="output/dota_lsknet_giou"  # The output directory to save logs and checkpoints
CONFIG_FILE_PATH="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"  # The detectron2 config file for R50-FPN Faster-RCNN
iSAID_DATASET_PATH="/apps/local/shared/CV703/datasets/iSAID/iSAID_patches"  # Path to iSAID dataset

# Create the output directory if not exists already
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

def register_dataset(path):
    register_coco_instances("iSAID_train", {},
                            f"{path}/train/instancesonly_filtered_train.json",
                            f"{path}/train/images/")
    register_coco_instances("iSAID_val", {},
                            f"{path}/val/instancesonly_filtered_val.json",
                            f"{path}/val/images/")

register_dataset(iSAID_DATASET_PATH)

def prepare_config(config_path, **kwargs):
    # Parse the expected key-word arguments
    output_path = kwargs["output_dir"]
    workers = kwargs["workers"]

    # Create and initialize the config
    cfg = get_cfg()
    cfg.SEED = 26911042  # Fix the random seed to improve consistency across different runs
    cfg.OUTPUT_DIR = output_path
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    # cfg.MODEL.WEIGHTS ='imagenet_fused_model.pth'

    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path) 
    cfg.DATASETS.TRAIN = ("iSAID_train",)
    cfg.DATASETS.TEST = ("iSAID_val",)
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
    cfg.MODEL.BACKBONE.NAME = 'build_lsknet_fpn_backbone'

    cfg.MODEL.BACKBONE.FREEZE_AT = 0 
    # Training schedule - equivalent to 0.25x schedule as per Detectron2 criteria
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"


    cfg.MODEL.WEIGHTS ='pretrained_weights/dota_lsknet_fused.pth'

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.STEPS = (60000, 80000)
    cfg.SOLVER.MAX_ITER = 100000

    return cfg

d2_config = prepare_config(CONFIG_FILE_PATH, output_dir=OUTPUT_DIRECTORY, workers=2)

# print(d2_config.dump())

# The trainer will automatically perform validation after training
class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return DatasetEvaluators([COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)])


trainer = Trainer(d2_config)  # Create Trainer
trainer.resume_or_load(resume=False)  # Set resume=True if intended to automatically resume training
trainer.train()  # Train and evaluate the model