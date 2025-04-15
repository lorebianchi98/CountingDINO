import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run convolutional_counting.py for multiple model configurations.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--run_from', type=int, default=0, help='Start the model evaluation from this model number in the list')
    args = parser.parse_args()

    # Set environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # List of model names
    models = [
        'dino_resnet50',
        'dino_vits8',
        'dino_vitb8',
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vits14_reg',
        'dinov2_vitb14_reg',
        'dinov2_vitl14_reg',
        'timm/vit_base_patch16_clip_224.openai', 
        'timm/vit_large_patch14_clip_224.openai', 
        'timm/vit_base_patch16_224.mae',
        'timm/vit_large_patch16_224.mae',
    ]
    
    official_models = [
        'dinov2_vitl14_reg',
    ]
    # python run.py --gpu 4 --run_from 8
    models = models[args.run_from:]

    # Base command
    base_command = ['python', 'convolutional_counting.py']

    # Loop through models and divide_et_impera settings
    count = 0
    for model in models:
        for split in ['test', 'val']:
            for ellipse_normalization in [True, False]:
                for ellipse_kernel_cleaning in [False, True]:
                    if ellipse_normalization != ellipse_kernel_cleaning:
                        continue
                    for divide_et_impera in [False, True]:
                        for divide_et_impera_twice in [False, True]:
                            if not divide_et_impera and divide_et_impera_twice:
                                continue
                            for filter_background in [False, True]:
                                # TABLE other backbone
                                if model not in official_models:
                                    if not ellipse_normalization:
                                        continue
                                    if not filter_background:
                                        continue
                                    if not divide_et_impera_twice:
                                        continue
                                # TABLE ellipse
                                if not ellipse_normalization:
                                    if not filter_background:
                                        continue
                                    if not divide_et_impera_twice:
                                        continue
                                # TABLE thresholding
                                if not filter_background:
                                    if not ellipse_normalization:
                                        continue
                                    if not divide_et_impera_twice:
                                        continue
                                
                                # TABLE divide et impera
                                if not divide_et_impera_twice:
                                    if not ellipse_normalization:
                                        continue
                                    if not filter_background:
                                        continue
                                    
                                    
                                
                                
                
                                command = base_command + [
                                    '--model_name', model,
                                    '--divide_et_impera', str(divide_et_impera),
                                    '--divide_et_impera_twice', str(divide_et_impera_twice),
                                    '--filter_background', str(filter_background),
                                    '--ellipse_normalization', str(ellipse_normalization),
                                    '--ellipse_kernel_cleaning', str(ellipse_kernel_cleaning),
                                    '--split', split,
                                ]
                                print(f"\nRunning: {' '.join(command)} on GPU {args.gpu}\n")
                                count += 1
                                subprocess.run(command)
    print(f"Runned {count} processes")
if __name__ == '__main__':
    main()
