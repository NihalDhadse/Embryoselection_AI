import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models, transforms
class VGGFeatureExtractor(nn.Module):
    def __init__(self, proj_dim=256, pretrained=True):
        super().__init__()
        # Load VGG16 with weights
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16(weights=weights)

        # Keep only the feature extraction convolutional backbone
        self.features = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Project 512-dimensional output down to standard projection dimension
        self.projection = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        return x

class SwinFeatureExtractor(nn.Module):
    def __init__(self, proj_dim=256, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        # Load Hierarchical Swin Transformer
        self.swin = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.swin.num_features

        # Project hierarchical features down to projection dimension
        self.projection = nn.Sequential(
            nn.Linear(num_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # timm Swin models return pooled output when num_classes=0
        x = self.swin(x)
        x = self.projection(x)
        return x
    
    
    
class HybridVGGSwinClassifier(nn.Module):
    
    def __init__(self, num_classes=7, proj_dim=256, dropout_rate=0.4, pretrained=True):
        super().__init__()
        self.vgg_branch = VGGFeatureExtractor(proj_dim=proj_dim, pretrained=pretrained)
        self.swin_branch = SwinFeatureExtractor(proj_dim=proj_dim, pretrained=pretrained)

        # Combined concatenated feature dimension is 2 * proj_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(2 * proj_dim, num_classes)
        )

    def forward(self, x):
        vgg_feats = self.vgg_branch(x)
        swin_feats = self.swin_branch(x)

        # Feature Fusion
        fused_feats = torch.cat((vgg_feats, swin_feats), dim=1)
        logits = self.classifier(fused_feats)
        return logits