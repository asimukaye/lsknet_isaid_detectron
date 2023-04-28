import fiftyone as fo

# Load a dataset stored in COCO format
dataset = fo.Dataset.from_dir(
    dataset_dir="/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/",
    data_path='train',
    dataset_type=fo.types.COCODetectionDataset,
    name='isaid'
)

# Explore the dataset in the App
session = fo.launch_app(dataset)