import argparse
import json
import os
import time
import multiprocessing as mp
from tqdm import tqdm
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import sys

# Setup paths for third-party dependencies
sys.path.extend([
    'third_party',
    './third_party/CutLER/cutler/demo',
    './third_party/CutLER/cutler'
])

from config import add_cutler_config
from predictor import VisualizationDemo

def setup_cfg(config_file, opts, confidence_threshold):
    """Load and configure the model."""
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    
    if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run CutLER inference on FSC147 dataset.")
    parser.add_argument('--output_path', type=str, default='predictions/preds_FSC147_val.json', help='Path to save predictions')
    parser.add_argument('--model_weights', type=str, default='third_party/CutLER/cutler/cutler_cascade_final.pth', help='Path to model weights')
    parser.add_argument('--splits_file', type=str, default='annotations/Train_Test_Val_FSC_147.json', help='Path to dataset split file')
    parser.add_argument('--split', type=str, default='test', help='Split to use')
    parser.add_argument('--confidence_threshold', type=float, default=0.05, help='Confidence threshold for predictions')
    parser.add_argument('--config_file', type=str, default='third_party/CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_demo.yaml', help='Path to model config file')
    parser.add_argument('--img_dir', type=str, default='/raid/datasets/FSC147/images_384_VarV2', help='Path to image directory')
    parser.add_argument('--save_every_n_steps', type=int, default=None, help='If is not None, every n steps the model will save the prediction file at output_path')
    return parser.parse_args()

def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    
    opts = ['MODEL.WEIGHTS', args.model_weights]
    cfg = setup_cfg(args.config_file, opts, args.confidence_threshold)
    demo = VisualizationDemo(cfg)
    
    if args.splits_file is not None:
        print("Loading dataset splits...")
        with open(args.splits_file, 'r') as f:
            splits = json.load(f)
        img_filenames = splits[args.split]
    else:
        img_filenames = os.listdir(args.img_dir)
    
    predictions = []
    print("Starting inference...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    for idx, img_filename in enumerate(tqdm(img_filenames)):
        path = os.path.join(args.img_dir, img_filename)
        img = read_image(path, format="BGR")
        height, width = img.shape[:2]

        img_predictions, _ = demo.run_on_image(img)

        predictions.append({
            'img_id': img_filename.split('.')[0],
            'width': width,
            'height': height,
            'bboxes': [bbox.tolist() for bbox in img_predictions['instances'].pred_boxes],
            'scores': img_predictions['instances'].scores.tolist(),
        })
        if args.save_every_n_steps is not None and idx > 0 and (idx % args.save_every_n_steps == 0):
            with open(args.output_path, 'w') as f:
                json.dump(predictions, f)
        
    
    with open(args.output_path, 'w') as f:
        json.dump(predictions, f)
    
    print(f"Predictions saved at {args.output_path}")

if __name__ == "__main__":
    main()