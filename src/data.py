import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SemanticStackDataset(Dataset):
    """
    Dataset for loading semantic stack tensors (house, tree, risk) for CNN training.
    """
    def __init__(self, file_list, transform=None):
        self.files = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path)
        
        # Extract Target from filename (Format: ID_TARGET.npy)
        try:
            target_str = filename.split('_')[-1].split('.')[0]
            target = int(target_str)
        except:
            target = 0 # Fallback
            
        # Load Tensor (H, W, C)
        img_array = np.load(path)
        
        # Normalize to 0-1 float
        img_array = img_array.astype(np.float32) / 255.0
        
        # Permute to (C, H, W) for PyTorch
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Apply Augmentations
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return img_tensor, torch.tensor(target, dtype=torch.float32).unsqueeze(0)

class WildfireDataset(Dataset):
    """
    Dataset for tabular data used in MLP training.
    """
    def __init__(self, X_data, y_data):
        self.X = torch.FloatTensor(X_data)
        self.y = torch.FloatTensor(y_data).unsqueeze(1)
        
    def __len__(self): return len(self.X)
    
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

