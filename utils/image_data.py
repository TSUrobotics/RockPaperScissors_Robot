import numpy as np
import torch

def extract_image_data(loader):
    X = []
    y = []
    for images, targets in loader:
        X.append(images.view(images.size(0), -1))
        y.append(targets)

    return torch.cat(X).numpy(), torch.cat(y).numpy()
