import json
import os
import numpy as np
import argparse
from sklearn.metrics import r2_score

def compute_metrics(preds, density_map_dir, cutler_conf_thresh, dino_conf_thresh):
    targets = []
    predictions = []

    for pred in preds:
        density_map = np.load(os.path.join(density_map_dir, f"{pred['img_id']}.npy"))
        
        # Target value: sum of density map
        target = density_map.sum()
        
        # Predicted value: count based on confidence thresholds
        predicted = sum([1 for cutler_conf, dino_conf in zip(pred['scores'], pred['exemplar_sims'])
                         if cutler_conf >= cutler_conf_thresh and dino_conf >= dino_conf_thresh])

        # Store values for evaluation
        targets.append(target)
        predictions.append(predicted)

    # Convert to numpy arrays for calculations
    targets = np.array(targets)
    predictions = np.array(predictions)

    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100  # Avoid division by zero
    r2 = r2_score(targets, predictions)

    # Return the results
    return {
        "MAE": mae,
        "MSE": mse,
        "MAPE": mape,
        "R2": r2
    }

def main(args):
    # Load predictions
    with open(args.pred_path, 'r') as f:
        preds = json.load(f)
    
    # Compute metrics
    metrics = compute_metrics(preds, args.density_map_dir, args.cutler_conf_thresh, args.dino_conf_thresh)
    
    # Print the metrics
    print(metrics)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Compute prediction evaluation metrics")
    
    # Arguments
    parser.add_argument('--pred_path', type=str, default='predictions/preds_FSC147.json', help="Path to the predictions JSON file")
    parser.add_argument('--density_map_dir', type=str, default='/raid/datasets/FSC147/gt_density_map_adaptive_384_VarV2', help="Directory containing the density map files")
    parser.add_argument('--cutler_conf_thresh', type=float, default=0.05, help="Confidence threshold for Cutler predictions")
    parser.add_argument('--dino_conf_thresh', type=float, default=0.5, help="Confidence threshold for Dino predictions")

    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    main(args)
