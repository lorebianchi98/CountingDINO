import argparse
import torch
import json
import os
import math
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# Import necessary utilities
from src.utils import convert_4corners_to_x1y1x2y2, adjust_bbox_for_transform, extract_bboxes_feats


def get_patch_feats(module, input, output):
    """Hook function to store the model output."""
    feats['outs'] = output


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print(f"Predictions updated and saved at {file_path}")


def extract_features(model, image_transforms, preds, examples, img_path, resize_dim, batch_size, device):
    """Extracts feature embeddings for predicted bounding boxes."""
    print("Starting feature extraction...")
    n_imgs = len(preds)
    n_batch = math.ceil(n_imgs / batch_size)

    for i in tqdm(range(n_batch)):
        start = i * batch_size
        end = start + batch_size if i < n_batch - 1 else n_imgs
        batch_size_ = end - start

        raw_imgs, example_bboxes, pred_bboxes = [], [], []
        
        for j in range(start, end):
            img_filename = f"{preds[j]['img_id']}.jpg"
            pred = preds[j]

            pil_img = Image.open(os.path.join(img_path, img_filename))
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            raw_imgs.append(pil_img)

            # Retrieve example bounding boxes
            example = examples[img_filename]
            h, w = pred['height'], pred['width']
            ex_bboxes = [convert_4corners_to_x1y1x2y2(bbox) for bbox in example['box_examples_coordinates']]
            ex_bboxes = [adjust_bbox_for_transform(w, h, bbox, resize_dim, resize_dim) for bbox in ex_bboxes]
            example_bboxes.append(ex_bboxes[:3]) # TODO: add dummy boxes to consider cases with more than 3 examples

            # Retrieve predicted bounding boxes
            pred_bbox = [adjust_bbox_for_transform(w, h, bbox, resize_dim, resize_dim) for bbox in pred['bboxes']]
            pred_bboxes.append(pred_bbox)

        batch_imgs = torch.stack([image_transforms(img) for img in raw_imgs]).to(device)

        with torch.no_grad():
            model.forward(batch_imgs)

        outs = feats['outs']

        # Extract patch embeddings
        patch_embeds = outs[:, 1:]
        example_bboxes = torch.tensor(example_bboxes)
        example_feats = extract_bboxes_feats(patch_embeds, example_bboxes)
        example_feats = example_feats.mean(dim=1)  # Aggregate example features per image

        # Prepare predicted bounding boxes
        n_max_boxes = max(map(len, pred_bboxes))
        new_bboxes = [pred_bbox + [[-1, -1, -1, -1]] * (n_max_boxes - len(pred_bbox)) if len(pred_bbox) < n_max_boxes else pred_bbox for pred_bbox in pred_bboxes]
        pred_bboxes = torch.tensor(new_bboxes)

        pred_feats = extract_bboxes_feats(patch_embeds, pred_bboxes)

        # L2 normalization
        pred_feats /= torch.norm(pred_feats, p=2, keepdim=True, dim=-1)
        example_feats /= torch.norm(example_feats, p=2, keepdim=True, dim=-1)

        # Compute similarity scores
        sims = torch.matmul(pred_feats, example_feats.unsqueeze(-1)).squeeze(-1)

        # Store results
        for j, sim in zip(range(start, end), sims):
            pred = preds[j]
            pred['exemplar_sims'] = sim[~sim.isnan()].tolist()

    return preds


def main():
    parser = argparse.ArgumentParser(description="Extract feature embeddings for bounding boxes using DINO.")
    parser.add_argument("--annotation_path", type=str, default="annotations/annotation_FSC147_384.json", help="Path to the annotation JSON file.")
    parser.add_argument("--pred_path", type=str, default="predictions/preds_FSC147.json", help="Path to the predictions JSON file.")
    parser.add_argument("--img_path", type=str, default="/raid/datasets/FSC147/images_384_VarV2", help="Path to the image dataset directory.")
    parser.add_argument("--model", type=str, default="dino_vitb8", help="Model to use")
    parser.add_argument("--resize_dim", type=int, default=480, help="Image resize dimension for feature extraction.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing images.")

    args = parser.parse_args()

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained DINO model
    model = torch.hub.load('facebookresearch/dino:main', args.model).to(device).eval()

    # Define image transformations
    image_transforms = T.Compose([
        T.Resize((args.resize_dim, args.resize_dim), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Register forward hook for feature extraction
    global feats
    feats = {}
    model.norm.register_forward_hook(get_patch_feats)

    # Load annotations and predictions
    examples = load_json(args.annotation_path)
    preds = load_json(args.pred_path)

    # Run feature extraction
    updated_preds = extract_features(model, image_transforms, preds, examples, args.img_path, args.resize_dim, args.batch_size, device)

    # Save updated predictions
    save_json(updated_preds, args.pred_path)


if __name__ == "__main__":
    main()
