import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

class RadarSwinTransformer(nn.Module):
    def __init__(self, num_classes=5, in_channels=1):
        super().__init__()
        # Load a pre-built Swin Transformer (Tiny version for edge efficiency)
        self.swin = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        
        # Modify the first layer to accept 1-channel (spectrogram) instead of 3-channel (RGB)
        first_layer = self.swin.features[0][0]
        self.swin.features[0][0] = nn.Conv2d(
            in_channels, 
            first_layer.out_channels, 
            kernel_size=first_layer.kernel_size, 
            stride=first_layer.stride, 
            padding=first_layer.padding
        )
        
        # Modify the classification head for our specific number of classes
        in_features = self.swin.head.in_features
        self.swin.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Input x should be a tensor of shape (batch_size, in_channels, H, W)
        return self.swin(x)
