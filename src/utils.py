import argparse
import itertools
import json
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F

from copy import deepcopy
from PIL import ImageDraw, Image
from sklearn.metrics import r2_score



def adjust_bbox_for_transform(orig_width, orig_height, bbox, target_width, target_height):
    """
    - Does not preserve the image scale.
    Adjusts the bounding box for an image resized to a fixed width and height.

    Args:
        bbox (list): The bounding box in [x1, y1, w, h] format.
        target_width (int): The width of the resized image.
        target_height (int): The height of the resized image.

    Returns:
        list: The adjusted bounding box in [x1, y1, w, h] format.
    """
    x1, y1, x2, y2 = bbox

    # Calculate scale factors for width and height
    scale_w = target_width / orig_width
    scale_h = target_height / orig_height

    # Adjust the bounding box
    x1 = x1 * scale_w
    y1 = y1 * scale_h
    x2 = x2 * scale_w
    y2 = y2 * scale_h

    # Return the adjusted bounding box
    return [x1, y1, x2, y2]


def draw_bounding_boxes(input_image, bounding_boxes, captions=[""], color="red", width=2, text_background=True, boxes_to_show = None):
    """
    Draws bounding boxes on an image.

    Args:
        image (PIL.Image): The image to draw on.
        bounding_boxes (list): A list of bounding boxes, each as [x1, y1, x2, y2].
        color (str): The color of the bounding boxes (default is red).
        width (int): The width of the bounding box lines (default is 2).

    Returns:
        PIL.Image: The image with bounding boxes drawn.
    """
    # Create a drawing context
    image = deepcopy(input_image)
    draw = ImageDraw.Draw( image )

    #scale = 720.0 / max(image.size)
    if boxes_to_show is not None:
        if isinstance(boxes_to_show, int):
            indexes_to_show = random.sample(range(len(bounding_boxes)), boxes_to_show)
        else:
            indexes_to_show = boxes_to_show
    
    for i, (bbox, cap ) in enumerate(itertools.zip_longest(bounding_boxes, captions, fillvalue="")):

        if boxes_to_show is not None:
            if i not in indexes_to_show: continue
        #bbox = [ i / scale for i in bbox ]
        #x1, y1, w, h = bbox
        x1, y1, x2, y2 = bbox
        
        #x2, y2 = x1 + w, y1 + h  # Convert width/height to bottom-right corner
        try:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            if cap != "":
                if text_background:
                    left,top,right,bottom = draw.multiline_textbbox((x1,y1), cap) #textbbox
                    draw.rectangle((left-5, top-5, right+5, bottom+5), fill="white")
                draw.multiline_text((x1,y1), cap, fill=color)   #text
            
        except Exception as e:
            print("exception, i: ", i, f"{x1 = } {y1 = } {x2 = }, {y2 = }")
            print(e)
    
    return image

def convert_4corners_to_x1y1x2y2(bbox):
    """
    Convert a bounding box from four-corner format to (x1, y1, x2, y2) format.
    
    Args:
        box (list): A list of four [x, y] points in the format 
                    [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]

    Returns:
        tuple: (x1, y1, x2, y2) where 
               x1, y1 = top-left corner
               x2, y2 = bottom-right corner
    """
    x1, y1 = bbox[0]  # Top-left corner
    x2, y2 = bbox[2]  # Bottom-right corner

    return x1, y1, x2, y2

def process_bboxes(imgs, bboxes, transform):
    transformed_bboxes = []
    bboxes = bboxes.tolist()
    for img, img_bboxes in zip(imgs, bboxes):
        for bbox in img_bboxes:
            # Crop the region defined by bbox
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
            cropped_region = img.crop((x_min, y_min, x_max, y_max))
            
            # Apply the transform to the cropped region
            transformed_region = transform(cropped_region)
            transformed_bboxes.append(transformed_region)
    
    return torch.stack(transformed_bboxes)

# Taken from https://github.com/Ruggero1912/Patch-ioner/blob/main/Patch-ioner/src/bbox_utils.py
def extract_bboxes_feats(patch_embeddings, bboxes, gaussian_avg=False, 
                         gaussian_bbox_variance=0.5, get_single_embedding_per_image=False,
                         patch_size=14, attention_map=None):
    """
    if get_single_embedding_per_image is True, the weights of all the bounding boxes patches on an image will be summed and the function will return the patch weights depending on this map
    """
    N = patch_embeddings.shape[0]
    N_boxes = bboxes.shape[1]
    grid_size = int(patch_embeddings.shape[1]**0.5)
    device = patch_embeddings.device

    bboxes //= patch_size
    bboxes = bboxes.int()

    # Reshape patches to grid
    patch_embeddings = patch_embeddings.view(N, grid_size, grid_size, -1)  # Shape (N, grid_size, grid_size, embed_dim)
    if attention_map is not None:
        attention_map = attention_map.view(N, grid_size, grid_size)  # Shape (N, grid_size, grid_size)
    # Grid of the sum of the gaussian weights
    total_patch_weights = torch.zeros(N, grid_size, grid_size)

    # Extract boxes
    x1, y1, w, h = bboxes.unbind(-1)  # Separate box dimensions (N, N_boxes)

    # Create mesh grid for slicing
    x2 = x1 + w  # Exclusive end x
    y2 = y1 + h  # Exclusive end y

    means = []
    for i in range(N):
        image_means = []
        for j in range(N_boxes):
            # if bboxes[i, j].sum().item() < 0:
            #     # this is the case where we receive a dummy box
            #     continue
            # Extract the region for each box
            region_patches = patch_embeddings[i, y1[i, j]:y2[i, j] + 1, x1[i, j]:x2[i, j] + 1, :]  # (h, w, embed_dim)
            
            if attention_map is not None:
                patch_weights = attention_map[i, y1[i, j]:y2[i, j] + 1, x1[i, j]:x2[i, j] + 1]
                patch_weights /= patch_weights.sum()
                total_patch_weights[i, y1[i, j]:y2[i, j] + 1, x1[i, j]:x2[i, j] + 1] += patch_weights
                
                weighted_patches = region_patches * patch_weights.to(device).unsqueeze(-1)  # (h, w, embed_dim)
                region_mean = weighted_patches.sum(dim=(0, 1))  # Weighted mean
                
            elif gaussian_avg:
                # Create Gaussian weights
                h_span, w_span = region_patches.shape[:2]
                y_coords, x_coords = torch.meshgrid(
                    torch.linspace(-1, 1, h_span),
                    torch.linspace(-1, 1, w_span),
                    indexing="ij"
                )
                if gaussian_bbox_variance == 0:
                    patch_weights = torch.zeros((h_span, w_span))
                    # Determine central indices
                    center_y = [h_span // 2] if h_span % 2 == 1 else [h_span // 2 - 1, h_span // 2]
                    center_x = [w_span // 2] if w_span % 2 == 1 else [w_span // 2 - 1, w_span // 2]
                    # Randomly select one of the central elements in even case
                    center_y = random.choice(center_y)
                    center_x = random.choice(center_x)
                    # Set the selected central element to 1
                    patch_weights[center_y, center_x] = 1.0
                else:
                    distances = x_coords**2 + y_coords**2
                    patch_weights = torch.exp(-distances / gaussian_bbox_variance)
                    patch_weights = patch_weights / patch_weights.sum()  # Normalize to sum to 1

                # Apply Gaussian weights to region patches
                weighted_patches = region_patches * patch_weights.to(device).unsqueeze(-1)  # (h, w, embed_dim)
                region_mean = weighted_patches.sum(dim=(0, 1))  # Weighted mean
                
                # Recording the bbox weight inside the image patch weight map
                total_patch_weights[i, y1[i, j]:y2[i, j] + 1, x1[i, j]:x2[i, j] + 1] += patch_weights
            else:
                # Mean pooling case: create uniform weights
                h_span, w_span = region_patches.shape[:2]
                uniform_weights = torch.ones(h_span, w_span) / (h_span * w_span)
                
                # Update total_patch_weights for mean pooling
                total_patch_weights[i, y1[i,j]:y2[i,j]+1, x1[i,j]:x2[i,j]+1] += uniform_weights
                
                # Compute mean of the region
                region_mean = region_patches.mean(dim=(0, 1))

            # Store the mean
            image_means.append(region_mean)
        if not get_single_embedding_per_image:
            means.append(torch.stack(image_means))

    # Normalizing the weight map so the sum is equal to 1
    total_patch_weights /= total_patch_weights.sum(dim=(1,2), keepdim=True)
    if not get_single_embedding_per_image:
        return torch.stack(means)  # Shape (N, N_boxes, embed_dim)
    else:
        # Expand dimensions to match embeddings
        total_patch_weights = total_patch_weights.unsqueeze(-1).to(device)

        # Compute weighted sum
        weighted_patch_mean = (total_patch_weights * patch_embeddings).sum(dim=(1, 2))  
        return  weighted_patch_mean 
    

def get_counting_metrics(predictions, targets, compute_madiff=False):
    # Convert to numpy arrays for calculations
    targets = np.array(targets)
    predictions = np.array(predictions)

    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100  # Avoid division by zero
    r2 = r2_score(targets, predictions)

    # Store results in a dictionary
    metrics = {
        "MAE": mae.item(),
        "RMSE": np.sqrt(mse).item(),
        "MAPE": mape.item(),
        "R2": r2
    }
    if compute_madiff:
        madiff = np.mean((predictions - targets))
        metrics["MADiff"] = madiff.item()
    
    return metrics

def log_results(data_dict, method_name, path='results/results.csv'):
    # Check if the file exists
    if os.path.exists(path):
        # If file exists, load the existing CSV file
        df = pd.read_csv(path)
    else:
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame(columns=['Method Name'] + list(data_dict.keys()))

    # Create a new row with the method_name and dictionary values
    new_row = pd.Series([method_name] + list(data_dict.values()), index=df.columns)

    # Append the new row using pd.concat instead of append
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(path, index=False)

    print("Row added")

def show_results(path='results/results.csv', n_digit=2):
    # Check if the file exists
    if os.path.exists(path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(path)
        # Show the DataFrame (print it)
        df = df.round(n_digit)
        return df
    else:
        print(f"File '{path}' does not exist.")
        
def exist_match_df(filename, filter_dict):
    df = pd.read_csv(filename)
    df = df.rename(columns={'Method Name': 'model_name'})
    # Only keep keys that exist in the DataFrame

    # Check for matching rows
    # matches = df[list(filter_dict_existing)].eq(pd.Series(filter_dict_existing)).all(axis=1)
    # Handle None values properly
    mask = pd.Series([True] * len(df))
    for k, v in filter_dict.items():
        if k in df.columns:
            if v is None:
                mask &= df[k].isna()
            else:
                mask &= df[k] == v

    any_match = mask.any()
    
    return bool(any_match)

def str2bool(value):
    """Convert 'True'/'False' to boolean."""
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
    

def create_feature_pyramide(feat1, feat2):
    # Upsample feat2 to match feat1's spatial dimensions
    feat2_upsampled = F.interpolate(feat2, size=feat1.shape[-2:], mode='bilinear', align_corners=False)

    # Concatenate along the channel dimension
    pyramid_feature = torch.cat([feat1, feat2_upsampled], dim=1)  
    
    return pyramid_feature

def rescale_tensor(tensor):
    """
    Rescale a tensor to the range [0, 1].
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def find_local_maxima(tensor, kernel_size=(3, 3)):
    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)
    kh, kw = kernel_size  # Unpack kernel size
    assert kh % 2 == 1 and kw % 2 == 1, "Kernel dimensions must be odd for proper centering"

    padding_h = (kh - 1) // 2
    padding_w = (kw - 1) // 2

    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
    max_pool = F.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=(padding_h, padding_w))

    local_maxima = (tensor == max_pool) & (tensor > 0)
    return local_maxima.squeeze(0).squeeze(0)  # Remove batch & channel dims

def resize_conv_maps(conv_maps):
    """
    Resize a list of convolutional maps to the size of the largest map in the list.
    """
    max_h, max_w = (0, 0) 
    resize_ratios = []
    for conv_map in conv_maps:
        max_h = max(max_h, conv_map.shape[1])
        max_w = max(max_w, conv_map.shape[2])

    resized_conv_maps = []
    for conv_map in conv_maps:
        orig_h, orig_w = conv_map.shape[-2:]
        resize_ratios.append((max_h / orig_h, max_w / orig_w))
        resized_conv_maps.append(F.interpolate(conv_map.unsqueeze(1), size=(max_h, max_w), mode='bilinear', align_corners=False).squeeze(1))
    resized_conv_maps = torch.cat(resized_conv_maps, dim=0)
    return resized_conv_maps, resize_ratios

def collapse_sizes(sizes):
    """
    Returns a tuple made of the average of the first and second elements of the input list of tuples.
    The input list of tuples is expected to be of the form [(h1, w1), (h2, w2), ...].
    """
    # Calculate the averages
    avg_first = sum(t[0] for t in sizes) / len(sizes)
    avg_second = sum(t[1] for t in sizes) / len(sizes)

    # Resulting tuple
    return (avg_first, avg_second)

def closest_odd_numbers(t):
    def closest_odd(x):
        if x < 3:
            return 3
        # Round to the nearest integer
        rounded = round(x)
        
        # If rounded number is even, adjust to the nearest odd number
        if rounded % 2 == 0:
            return rounded + 1 if x >= rounded else rounded - 1
        return rounded
    
    return tuple(closest_odd(x) for x in t)

def split_image_into_four(image: Image.Image):
    """Splits a PIL image into 4 equal parts and returns a list of the 4 images."""
    width, height = image.size
    mid_x, mid_y = width // 2, height // 2
    
    # Define the 4 regions (left-top, right-top, left-bottom, right-bottom)
    boxes = [
        (0, 0, mid_x, mid_y),         # Top-left
        (mid_x, 0, width, mid_y),      # Top-right
        (0, mid_y, mid_x, height),     # Bottom-left
        (mid_x, mid_y, width, height)  # Bottom-right
    ]
    
    # Crop and return images
    return [image.crop(box) for box in boxes]

def merge_feature_maps(feats):
    # Arrange feature maps in a 2x2 grid
    top_row = torch.cat([feats[0], feats[1]], dim=-1)  # Concatenate horizontally (width-wise)
    bottom_row = torch.cat([feats[2], feats[3]], dim=-1)  # Concatenate horizontally (width-wise)
    
    merged_map = torch.cat([top_row, bottom_row], dim=-2)  # Concatenate vertically (height-wise)
    
    return merged_map

def get_features(model, img, image_transforms, map_keys, divide_et_impera=False, divide_et_impera_twice=False):
    device = next(model.parameters()).device
    if divide_et_impera:
        crops = split_image_into_four(img)
        if divide_et_impera_twice:
            crops = [minicrop for crop in crops for minicrop in split_image_into_four(crop)]
        
        batch_imgs = torch.stack([image_transforms(crop) for crop in crops])
    else:
        batch_imgs = image_transforms(img).unsqueeze(0)
    with torch.no_grad():
        outs = model(batch_imgs.to(device))
        
    for n_map, map_key in enumerate(map_keys):
        if n_map == 0:
            feats = outs[map_key]
        else:
            feats = create_feature_pyramide(feats, outs[map_key])
    
    if divide_et_impera:
        # n_patch_x_row = feats.shape[-1]
        # feats = feats.view(1, feats.shape[1], n_patch_x_row * 2, n_patch_x_row * 2)
        if not divide_et_impera_twice:
            feats = merge_feature_maps(feats).unsqueeze(0)
        if divide_et_impera_twice:
            new_feats = []
            for i in range(4):
                new_feats.append(merge_feature_maps(feats[i * 4 : (i + 1) * 4]).unsqueeze(0))
            feats = torch.cat(new_feats, dim=0)
            feats = merge_feature_maps(feats).unsqueeze(0)
    
    return feats

def compute_avg_conv_filter(pooled_features_list):
    """
    Compute the average of a list of convolutional feature maps. The feature map could be of diffreent sizes.
    - pooled_features_list: list of feature maps, with shape (channels, height, width)
    """
    pooled_features_list = [feat.squeeze(0) for feat in pooled_features_list]
    # compute the average size of the feature maps
    avg_size_float = collapse_sizes([feat.shape[1:] for feat in pooled_features_list])
    avg_size_h, avg_size_w = int(avg_size_float[0]), int(avg_size_float[1])

    interpolated_features = []
    for pooled_feat in pooled_features_list:
        interpolated_feat = F.interpolate(pooled_feat.unsqueeze(1), size=(avg_size_h, avg_size_w), mode='bilinear', align_corners=False).squeeze(1)
        interpolated_features.append(interpolated_feat)
    # Concatenate the interpolated feature maps along the batch dimension
    interpolated_features = torch.stack(interpolated_features, dim=0)
    # Compute the average feature map
    avg_feature_map = torch.mean(interpolated_features, dim=0)
    return avg_feature_map
    #torch.mean(torch.cat(pooled_features_list, dim=0), dim=0)
    
def bboxes_tointeger(bboxes, remove_bbox_intersection=False):
    if not remove_bbox_intersection:
        bboxes = np.column_stack([np.floor(bboxes[:, 0]), np.floor(bboxes[:, 1]), 
                                    np.ceil(bboxes[:, 2]), np.ceil(bboxes[:, 3])]).astype(int)
    else:
        bboxes = np.column_stack([
            np.ceil(bboxes[:, 0]), 
            np.ceil(bboxes[:, 1]), 
            np.maximum(np.floor(bboxes[:, 2]), np.ceil(bboxes[:, 0]) + 1),  # Ensure width >= 1
            np.maximum(np.floor(bboxes[:, 3]), np.ceil(bboxes[:, 1]) + 1),  # Ensure height >= 1
        ]).astype(int)
    return bboxes

def rescale_bbox(bbox, output, feats):
    scale_y = output.shape[1] / feats.shape[2]
    scale_x = output.shape[2] / feats.shape[3]

    rescaled_bbox = bbox.clone().float()
    rescaled_bbox[0] *= scale_x  # x1
    rescaled_bbox[2] *= scale_x  # x2
    rescaled_bbox[1] *= scale_y  # y1
    rescaled_bbox[3] *= scale_y  # y2
    return rescaled_bbox


def ellipse_coverage(h, w, samples_per_pixel=10):
    # Create a grid of pixel centers
    y = torch.linspace(0.5, h - 0.5, h).view(h, 1).expand(h, w)
    x = torch.linspace(0.5, w - 0.5, w).view(1, w).expand(h, w)

    # Generate subpixel offsets for sampling
    sp = samples_per_pixel
    sub = torch.linspace(-0.5 + 1/(2*sp), 0.5 - 1/(2*sp), sp)
    dy, dx = torch.meshgrid(sub, sub, indexing='ij')
    offsets = torch.stack([dy.reshape(-1), dx.reshape(-1)], dim=1)  # (sp*sp, 2)

    # Expand pixel grid to sample subpixel points
    total_samples = sp * sp
    x_samples = x.unsqueeze(-1) + offsets[:, 1].view(1, 1, -1)
    y_samples = y.unsqueeze(-1) + offsets[:, 0].view(1, 1, -1)

    # Normalize coordinates to ellipse space
    norm_x = (x_samples - w / 2) / (w / 2)
    norm_y = (y_samples - h / 2) / (h / 2)

    inside = (norm_x ** 2 + norm_y ** 2) <= 1.0
    coverage = inside.float().mean(dim=2)  # average over samples

    return coverage

def exist_and_delete_match_df(filename, filter_dict):
    # exist_and_delete_match_df(args.log_file, row_params_dict)
    if not os.path.exists(filename):
        return False
    
    df = pd.read_csv(filename)
    df = df.rename(columns={'Method Name': 'model_name'})

    # Build the mask for matching rows
    mask = pd.Series([True] * len(df))
    for k, v in filter_dict.items():
        if k in df.columns:
            if v is None:
                mask &= df[k].isna()
            else:
                mask &= df[k] == v

    if mask.any():
        # Drop matching rows
        df = df[~mask]
        df.to_csv(filename, index=False)
        return True  # Match existed and was deleted
    else:
        return False  # No match found

def add_dummy_row(method_name, args_dict, path='results/results.csv'):
    scores_dummy = {
        'MAE': 0,
        'RMSE': 0,
        'MAPE': 0,
        'R2': 0,
    }
    data_dict = {**args_dict, **scores_dummy}
    df = pd.read_csv(path)
    new_row = pd.Series([method_name] + list(data_dict.values()), index=df.columns)
    idx = len(df) 
    df.loc[idx] = new_row
    df.to_csv(path, index=False)
    return idx

def delete_row(idx, path='results/results.csv'):
    df = pd.read_csv(path)
    df = df.drop(idx)
    df.to_csv(path, index=False)
    