from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import (
            COCOEvaluator, DatasetEvaluators
            )

import lsknet


OUTPUT_DIRECTORY = "output/imagenet_lsknet_t"
MODEL_CHECKPOINTS_PATH=f"{OUTPUT_DIRECTORY}/model_0089999.pth"

CONFIG_FILE_PATH="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"

def register_datase(path):
    register_coco_instances("iSAID_train", {},
                            f"{path}/train/instancesonly_filtered_train.json",
                            f"{path}/train/images/")
    register_coco_instances("iSAID_val", {},
                            f"{path}/val/instancesonly_filtered_val.json",
                            f"{path}/val/images/")

iSAID_DATASET_PATH="/apps/local/shared/CV703/datasets/iSAID/iSAID_patches"
register_datase(iSAID_DATASET_PATH)



def prepare_config(config_path, **kwargs):
    # Parse the expected key-word arguments
    output_path = kwargs["output_dir"]
    workers = kwargs["workers"]

    # Create and initialize the config
    cfg = get_cfg()
    cfg.SEED = 26911042  # Fix the random seed to improve consistency across different runs
    cfg.OUTPUT_DIR = output_path
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    # cfg.MODEL.WEIGHTS ='modified_model.pth'
    cfg.DATASETS.TRAIN = ("iSAID_train",)
    cfg.DATASETS.TEST = ("iSAID_val",)
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
    cfg.MODEL.BACKBONE.NAME = 'build_lsknet_t_fpn_backbone' 
    cfg.MODEL.BACKBONE.FREEZE_AT = 0 
    # Training schedule - equivalent to 0.25x schedule as per Detectron2 criteria
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.STEPS = (60000, 80000)
    cfg.SOLVER.MAX_ITER = 100000

    return cfg

d2_config = prepare_config(CONFIG_FILE_PATH, output_dir=OUTPUT_DIRECTORY, workers=2)
# Path to the trained ".pth" model file

# d2_config.MODEL.WEIGHTS = MODEL_CHECKPOINTS_PATH  # Update the weights path in the config
d2_config.OUTPUT_DIR = f"{OUTPUT_DIRECTORY}/validate_90k"  # Update the output directory path

setup_logger(f"{d2_config.OUTPUT_DIR}/log.txt")  # Setup the logger


# The trainer will automatically perform validation after training
class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return DatasetEvaluators([COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)])
    

model = Trainer.build_model(d2_config)  # Build model using Trainer
DetectionCheckpointer(model, save_dir=d2_config.OUTPUT_DIR).load(MODEL_CHECKPOINTS_PATH)  # Load the checkpoints
Trainer.test(d2_config, model)  # Test the model on the validation set