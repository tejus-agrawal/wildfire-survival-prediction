import torch
import torch.nn as nn
from torchvision import models

def get_resnet_model(device):
    """
    Returns a ResNet18 model modified for binary classification.
    """
    model = models.resnet18(weights=None) # Training from scratch on geometric shapes
    num_ftrs = model.fc.in_features
    
    # Custom Head for Binary Classification
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 64),
        nn.ReLU(),
        nn.Dropout(0.5), # Regularization
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model.to(device)

class DynamicMLP(nn.Module):
    """
    Dynamic Multi-Layer Perceptron with configurable layers and dropout.
    """
    def __init__(self, input_dim, layers=[128, 64], dropout=0.3):
        super().__init__()
        module_list = []
        prev_dim = input_dim
        
        for dim in layers:
            module_list.append(nn.Linear(prev_dim, dim))
            module_list.append(nn.BatchNorm1d(dim))
            module_list.append(nn.ReLU())
            module_list.append(nn.Dropout(dropout))
            prev_dim = dim
            
        module_list.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*module_list)
        
    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # pt is probability of correct classification
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

