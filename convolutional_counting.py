import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.ops as ops

import numpy as np
import re
import timm

from src.model import VisualBackbone
from src.utils import (
    convert_4corners_to_x1y1x2y2, 
    get_counting_metrics, 
    log_results,
    add_dummy_row,
    exist_match_df,
    exist_and_delete_match_df,
    load_json, 
    get_features, 
    bboxes_tointeger, 
    compute_avg_conv_filter, 
    rescale_tensor,
    resize_conv_maps,
    rescale_bbox,
    str2bool,
    ellipse_coverage
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def process_example(
    idx, img_filename, entry, model, transform, map_keys, img_dir, density_map_dir, config, return_maps=False
):
    img = Image.open(os.path.join(img_dir, img_filename)).convert('RGB')
    density_map = np.load(os.path.join(density_map_dir, f"{img_filename.split('.')[0]}.npy"))
    w, h = img.size    

    with torch.no_grad():
        feats = get_features(
            model, img, transform, map_keys,
            divide_et_impera=config.divide_et_impera,
            divide_et_impera_twice=config.divide_et_impera_twice
        )
        if config.cosine_similarity or config.normalize_features:
            feats = feats / feats.norm(dim=1, keepdim=True)

    # Process exemplars
    ex_bboxes = [convert_4corners_to_x1y1x2y2(b) for b in entry['box_examples_coordinates']]
    if config.num_exemplars is not None:
        assert config.num_exemplars > 0, "num_exemplars must be greater than 0. config.num_exemplars = " + config.num_exemplars
        ex_bboxes = ex_bboxes[:config.num_exemplars]
    bboxes = np.array([(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in ex_bboxes]) * feats.shape[-1]
    bboxes = bboxes_tointeger(bboxes, config.remove_bbox_intersection)

    conv_maps = []
    pooled_features_list = []
    output_sizes = []
    rescaled_bboxes = []

    for bbox in bboxes:
        bbox_tensor = torch.tensor(bbox)
        output_size = (
            int(bbox_tensor[3] - bbox_tensor[1]), 
            int(bbox_tensor[2] - bbox_tensor[0])
        )

        pooled = ops.roi_align(
            feats, [bbox_tensor.unsqueeze(0).float().to(device)],
            output_size=output_size, spatial_scale=1.0
        )
        if config.ellipse_kernel_cleaning:
            ellipse = ellipse_coverage(pooled.shape[-2], pooled.shape[-1]).unsqueeze(0).unsqueeze(0).to(device)
            pooled *= ellipse
            
        pooled_features_list.append(pooled)

        if config.exemplar_avg:
            continue

        conv_weights = pooled.view(feats.shape[1], 1, *output_size)
        conv_layer = nn.Conv2d(
            in_channels=feats.shape[1],
            out_channels=1 if config.cosine_similarity else feats.shape[1],
            kernel_size=output_size,
            padding=0,
            groups=1 if config.cosine_similarity else feats.shape[1],
            bias=False
        )
        conv_layer.weight = nn.Parameter(pooled if config.cosine_similarity else conv_weights)

        with torch.no_grad():
            output = conv_layer(feats[0])

        if config.correct_bbox_resize:
            rescaled_bbox = rescale_bbox(bbox_tensor, output, feats)
        else:
            rescaled_bbox = bbox_tensor

        rescaled_bboxes.append(rescaled_bbox)

        if config.use_roi_norm and not config.roi_norm_after_mean:
            if config.cosine_similarity:
                output += 1.0
            pooled_output = ops.roi_align(
                output.unsqueeze(0), [rescaled_bbox.unsqueeze(0).float().to(device)],
                output_size=output_size, spatial_scale=1.0
            )
            output = output / pooled_output.sum()

        conv_maps.append(output)
        output_sizes.append(output_size)

    if config.exemplar_avg:
        pooled = compute_avg_conv_filter(pooled_features_list)
        output_size = pooled.shape[1:]
        conv_weights = pooled.view(pooled.shape[0], 1, *output_size)

        conv_layer = nn.Conv2d(
            in_channels=feats.shape[1],
            out_channels=1 if config.cosine_similarity else feats.shape[1],
            kernel_size=output_size,
            padding=0,
            groups=1 if config.cosine_similarity else feats.shape[1],
            bias=False
        )
        conv_layer.weight = nn.Parameter(pooled.unsqueeze(0) if config.cosine_similarity else conv_weights)

        with torch.no_grad():
            output = conv_layer(feats[0])

        if config.use_roi_norm and not config.roi_norm_after_mean:
            raise NotImplementedError("ROI norm after conv_mean is not implemented for average-based filter.")

        conv_maps.append(output)
        output_sizes.append(output_size)

    output = post_process_density_map(
        conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config
    )
    if return_maps:
        return density_map, output
    return density_map.sum().item(), output.sum().item()


def post_process_density_map(conv_maps, pooled_feats, bboxes, output_sizes, config):
    if config.use_threshold:
        output, resize_ratios = resize_conv_maps(conv_maps)
        output = output.mean(dim=0)
        if config.use_minmax_norm:
            output = rescale_tensor(output)

        thresh = torch.median(output)
        output[output < thresh] = 0
        return output

    if config.use_roi_norm and config.roi_norm_after_mean:
        output, resize_ratios = resize_conv_maps(conv_maps)
        output = output.mean(dim=0)
        if config.use_minmax_norm:
            output = rescale_tensor(output)

        pooled_vals = []
        for bbox, ratio in zip(bboxes, resize_ratios):
            scaled_bbox = torch.tensor([
                bbox[0] * ratio[1], bbox[1] * ratio[0],
                bbox[2] * ratio[1], bbox[3] * ratio[0]
            ]).int()
            # scaled_bbox = torch.tensor(bboxes_tointeger(scaled_bbox.unsqueeze(0), config.remove_bbox_intersection)[0])
            output_size = (
                int(scaled_bbox[3] - scaled_bbox[1]),
                int(scaled_bbox[2] - scaled_bbox[0])
            )
            pooled = ops.roi_align(
                output.unsqueeze(0).unsqueeze(0),
                [scaled_bbox.unsqueeze(0).float().to(device)],
                output_size=output_size, spatial_scale=1.0
            )
            pooled_vals.append(pooled)

        if config.ellipse_normalization:
            norm_coeff = sum([(p[0, 0] * ellipse_coverage(p.shape[-2], p.shape[-1]).to(device)).sum() for p in pooled_vals]) / (len(pooled_vals) * config.scaling_coeff)
        else:
            norm_coeff = sum([p.sum() for p in pooled_vals]) / (len(pooled_vals) * config.scaling_coeff)
        if config.fixed_norm_coeff is not None:
            norm_coeff = config.fixed_norm_coeff

        output = output / norm_coeff
        if config.filter_background is True:
            thresh = max( [f.shape[-2] * f.shape[-1] for f in pooled_feats] )
            thresh = (1 / thresh ) * 1.0
            output[output < thresh] = 0

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dinov2_vitb14_reg')
    parser.add_argument('--img_dir', type=str, default='/raid/datasets/FSC147/images_384_VarV2')
    parser.add_argument('--density_map_dir', type=str, default='/raid/datasets/FSC147/gt_density_map_adaptive_384_VarV2')
    parser.add_argument('--annotation', type=str, default='annotations/annotation_FSC147_384.json')
    parser.add_argument('--splits', type=str, default='annotations/Train_Test_Val_FSC_147.json')
    parser.add_argument('--log_file', type=str, default='results/results.csv')
    
    parser.add_argument('--divide_et_impera', type=str2bool, default=False)
    parser.add_argument('--divide_et_impera_twice', type=str2bool, default=False)
    parser.add_argument('--exemplar_avg', type=str2bool, default=False)
    parser.add_argument('--cosine_similarity', type=str2bool, default=False)
    parser.add_argument('--normalize_features', type=str2bool, default=False)
    parser.add_argument('--normalize_only_biggest_bbox', type=str2bool, default=False)
    parser.add_argument('--use_threshold', type=str2bool, default=False)
    parser.add_argument('--use_roi_norm', type=str2bool, default=True)
    parser.add_argument('--roi_norm_after_mean', type=str2bool, default=True)
    parser.add_argument('--use_minmax_norm', type=str2bool, default=True)
    parser.add_argument('--remove_bbox_intersection', type=str2bool, default=False)
    parser.add_argument('--correct_bbox_resize', type=str2bool, default=True)
    parser.add_argument('--scaling_coeff', type=float, default=1.0)
    parser.add_argument('--fixed_norm_coeff', type=float, default=None)
    parser.add_argument('--filter_background', type=str2bool, default=False)
    parser.add_argument('--ellipse_normalization', type=str2bool, default=False)
    parser.add_argument('--ellipse_kernel_cleaning', type=str2bool, default=False)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_exemplars', type=int, default=None)

    parser.add_argument('--save_preds_to_file', type=str2bool, default=False)
    parser.add_argument('--log_results', type=str2bool, default=True)
    parser.add_argument('--no_skip', type=str2bool, default=False)
# CUDA_VISIBLE_DEVICES=4 python convolutional_counting.py --model_name dino_resnet50 --divide_et_impera True --divide_et_impera_twice True --filter_background False --ellipse_normalization True --ellipse_kernel_cleaning True --split test
    # 
    args = parser.parse_args()

    save_preds_to_file = args.save_preds_to_file

    row_params_dict = {k: v for k, v in vars(args).items() if k not in ['img_dir', 'density_map_dir', 'annotation', 'splits', 'log_file']}
    args_dict = {k: v for k, v in vars(args).items() if k not in ['model_name', 'img_dir', 'density_map_dir', 'annotation', 'splits', 'save_preds_to_file', 'log_results', 'no_skip', 'log_file']}
    if exist_match_df(args.log_file, row_params_dict) and not args.no_skip:
        #print("Already done for this configuration. Skipping...")
        return
    else:
        # create dummy row
        dummy_idx = add_dummy_row(args.model_name, args_dict, args.log_file)
        
    
    print("Parameters Recap:")
    print(json.dumps(vars(args), indent=4))
    

    if 'mae' in args.model_name or 'clip' in args.model_name or 'sam' in args.model_name:
        match = re.search(r'patch(\d+)', args.model_name)
        patch_size = int(match.group(1))
        resize_dim = patch_size * 60
        model = VisualBackbone(args.model_name, img_size=resize_dim).to(device).eval()
        data_config = timm.data.resolve_model_data_config(model)
        data_config['input_size'] = (3, resize_dim, resize_dim)
        transform = timm.data.create_transform(**data_config, is_training=False)
    else:
        resize_dim = 840 if 'dinov2' in args.model_name else 480
        model = VisualBackbone(args.model_name, img_size=resize_dim).to(device).eval()
        transform = T.Compose([
            T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    examples = load_json(args.annotation)
    splits = load_json(args.splits)
    examples = {k: v for k, v in examples.items() if k in splits[args.split]}

    map_keys = ['vit_out'] if 'vit' in args.model_name else ['map3']

    predictions, targets = [], []

    for idx, (img_filename, entry) in tqdm(enumerate(examples.items()), total=len(examples)):
        gt, pred = process_example(idx, img_filename, entry, model, transform, map_keys,
                                   args.img_dir, args.density_map_dir, args)
        predictions.append(pred)
        targets.append(gt)
        
    if save_preds_to_file:
        save_dir = os.path.join('results', args.model_name)
        os.makedirs(save_dir, exist_ok=True)
        # out file name must be specific to the configuration
        out_file_name = f"predictions_{args.model_name}_{'_'.join([f'{k}_{v}' for k, v in args_dict.items()])}.npy"
        np.save(os.path.join(save_dir, out_file_name), predictions)
        #np.save(os.path.join(save_dir, 'targets.npy'), targets)
    metrics = get_counting_metrics(predictions, targets)
    print(metrics)

    if args.log_results:
        # Logging the results
        exist_and_delete_match_df(args.log_file, row_params_dict)
        log_results({**args_dict, **metrics}, args.model_name, path=args.log_file)
        
if __name__ == '__main__':
    main()
