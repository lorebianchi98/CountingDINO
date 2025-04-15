import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as T
import torchvision
import timm

class VisualBackbone(nn.Module):
    def __init__(self, model_name, img_size=480):
        super(VisualBackbone, self).__init__()
        
        # Flag to check if the model is a Vision Transformer (ViT)
        self.is_vit = False
        self.is_dinov2 = False  # Used for distinguishing between DINO and DINOv2
        self.model_name = model_name
        
        # Choose the appropriate feature extractor based on the model name
        if 'dino' in model_name:
            self._setup_dino_model(model_name)
        elif 'mae' in model_name or 'clip' in model_name or 'sam' in model_name:
            self._setup_other_transformer_models(model_name, img_size)
        else:
            # Default to ResNet50 for non-transformer models
            self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        
        # If the model is a ResNet, split it into different convolutional blocks
        if 'resnet' in model_name:
            self._split_resnet_layers()
        
    def _setup_dino_model(self, model_name):
        """
        Setup for DINO models (ResNet or ViT-based).
        """
        if 'resnet' in model_name:
            # Load DINO with a ResNet backbone
            self.feature_extractor = torch.hub.load('facebookresearch/dino:main', model_name) 
        else:
            # Mark as Vision Transformer (ViT)
            self.is_vit = True
            if 'dinov2' in model_name:
                self.is_dinov2 = True
                # Load DINOv2 model
                self.feature_extractor = torch.hub.load('facebookresearch/dinov2', model_name) 
            else:
                # Load standard DINO ViT model
                self.is_dinov2 = False
                self.feature_extractor = torch.hub.load('facebookresearch/dino:main', model_name) 
                # Add hook to extract patch tokens from the feature extractor
                self.feature_extractor.norm.register_forward_hook(self._get_dino_patches_hook)

    def _setup_other_transformer_models(self, model_name, img_size):
        """
        Setup for transformer models like MAE, CLIP, and SAM using timm library.
        """
        self.is_vit = True
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classifier head
            img_size=img_size  # Resize images for the model
        )
        
    def _split_resnet_layers(self):
        """
        Split ResNet50 layers into different blocks.
        """
        children = list(self.feature_extractor.children())
        self.conv1 = nn.Sequential(*children[:4])  # Initial convolution layers
        self.conv2 = children[4]  # Layer after the first conv block
        self.conv3 = children[5]  # After second conv block
        self.conv4 = children[6]  # After third conv block

    def _get_dino_patches_hook(self, module, input, output):
        """
        Hook function to store patch tokens for DINO model.
        """
        module.patch_outs = output
            
            
    def forward(self, im_data):
        """
        Forward pass to extract features from input images.
        """
        feat = OrderedDict()

        # If not a ViT model (e.g., using ResNet)
        if not self.is_vit:
            feat_map = self.conv1(im_data)
            feat_map = self.conv2(feat_map)
            feat_map3 = self.conv3(feat_map)
            feat_map4 = self.conv4(feat_map3)
            feat['map3'] = feat_map3
            feat['map4'] = feat_map4
        # Handle SAM, CLIP, and MAE cases (transformers)
        elif 'mae' in self.model_name or 'clip' in self.model_name or 'sam' in self.model_name:
            # For SAM, CLIP, and MAE, use forward_features to get the features
            patch_feats = self.feature_extractor.forward_features(im_data)[:, 1:, :]

            # Store the extracted features in the 'vit_out' key
            feat['vit_out'] = patch_feats
        else:
            # If it's a ViT model (DINO or DINOv2)
            if self.is_dinov2:
                # Get patch features for DINOv2
                patch_feats = self.feature_extractor(im_data, is_training=True)['x_norm_patchtokens']
            else:
                # Get patch features for DINO ViT
                self.feature_extractor(im_data)  # Run through the model to activate hook
                patch_feats = self.feature_extractor.norm.patch_outs[:, 1:, :]  # Extract patch tokens
                
        if self.is_vit:
            # Convert 1D patch tokens to 2D spatial grid for ViT models
            n_patch_x_row = int(patch_feats.shape[1] ** 0.5)  # Assume square grid
            patch_feats = patch_feats.permute(0, 2, 1)  # Reorder dimensions
            patch_feats = patch_feats.view(patch_feats.shape[0], patch_feats.shape[1], n_patch_x_row, n_patch_x_row)  # Reshape to 2D grid
            feat['vit_out'] = patch_feats


        return feat
