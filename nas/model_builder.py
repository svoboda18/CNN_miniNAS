import torch
import torch.nn as nn
from typing import Dict, Any, List

class ModelBuilder:
    @staticmethod
    def calculate_flatten_size(architecture: Dict[str, Any]) -> int:
        """
        Calculates the input feature count for the linear layer by 
        simulating spatial dimensions through the conv/pool layers.
        """
        spatial = architecture.get('input_size', 28)
        current_channels = architecture.get('input_channels', 1)

        for layer in architecture['layers']:
            lt = layer['type']
            if lt == 'Conv2d':
                k = layer['kernel']
                p = layer.get('padding', 0)
                # Formula: [(W − K + 2P) / S] + 1 (Stride defaults to 1 here)
                spatial = (spatial - k + 2 * p) + 1
                current_channels = layer['channels']
            elif lt == 'MaxPool2d':
                k = layer['kernel']
                s = layer.get('stride', k)
                spatial = (spatial - k) // s + 1
        
        return current_channels * spatial * spatial

    @staticmethod
    def build_from_dict(architecture: Dict[str, Any]) -> nn.Module:
        """
        Dynamically builds the PyTorch model.
        Resolves the Pylance type errors by using a typed list of nn.Modules.
        """
        # --- Feature Extraction (CNN) ---
        feature_modules: List[nn.Module] = []
        in_ch = architecture.get('input_channels', 1)

        for l_cfg in architecture['layers']:
            lt = l_cfg['type']
            if lt == 'Conv2d':
                feature_modules.append(nn.Conv2d(in_ch, l_cfg['channels'], 
                                               kernel_size=l_cfg['kernel'], 
                                               padding=l_cfg.get('padding', 0)))
                in_ch = l_cfg['channels']
            elif lt == 'ReLU':
                feature_modules.append(nn.ReLU())
            elif lt == 'MaxPool2d':
                feature_modules.append(nn.MaxPool2d(kernel_size=l_cfg['kernel'], 
                                                  stride=l_cfg.get('stride', 2)))
            elif lt == 'Dropout':
                feature_modules.append(nn.Dropout(p=l_cfg.get('rate', 0.1)))

        # --- Classifier (MLP) ---
        flat_size = ModelBuilder.calculate_flatten_size(architecture)
        last_hid = architecture.get('last_hid_mlp', 0)
        num_classes = architecture.get('num_classes', 10)

        classifier_modules: List[nn.Module] = [nn.Flatten()]
        
        if last_hid > 0:
            classifier_modules.append(nn.Linear(flat_size, last_hid))
            classifier_modules.append(nn.ReLU())
            classifier_modules.append(nn.Linear(last_hid, num_classes))
        else:
            classifier_modules.append(nn.Linear(flat_size, num_classes))

        # Assemble into a standard nn.Module structure
        class NASModel(nn.Module):
            def __init__(self, features, classifier):
                super().__init__()
                self.features = nn.Sequential(*features)
                self.classifier = nn.Sequential(*classifier)
            
            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)

        return NASModel(feature_modules, classifier_modules)