from utils.arguments import set_args
from utils.extract_features import init_landmark_model, extract_landmark_features
from utils.get_paths import get_paths, load_yaml_config
from utils.get_torch_device import torch_device
from utils.YOLODatasetLoader import YOLODatasetLoader

import argparse
import os
import sys
from pathlib import Path
from types import NoneType

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def main():
    args = set_args()

    device = torch_device()

    # Transforms (Must match training & inference)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset paths
    cfg = load_yaml_config(args.data)
    print(cfg['names'])
    train_img, train_lbl = get_paths(cfg, args.data, 'train')
    test_img,  test_lbl  = get_paths(cfg, args.data, 'test')

    # Datasets & loaders
    train_set = YOLODatasetLoader(train_img, train_lbl, transforms=transform)
    test_set = YOLODatasetLoader(test_img,  test_lbl,  transforms=transform)

    train_loader = DataLoader(train_set,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    test_loader  = DataLoader(test_set, 
                              batch_size=32,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=True)

    current_dir = os.getcwd()
    model_path = f'{current_dir}/models/gesture_recognizer.task'

    landmark_model = init_landmark_model(model_path)

    # --- Feature Extraction (unchanged from previous fix) ---
    X_train, y_train = [], []
    for imgs, lbls in train_loader:
        x, y = extract_landmark_features(landmark_model, imgs, lbls)
        X_train.append(x); y_train.append(y)
    X_train, y_train = np.vstack(X_train), np.concatenate(y_train)

    X_test, y_test = [], []
    for imgs, lbls in test_loader:
        x, y = extract_landmark_features(landmark_model, imgs, lbls)
        X_test.append(x); y_test.append(y)
    X_test, y_test = np.vstack(X_test), np.concatenate(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    valid_mask_train = np.any(X_train != 0, axis=1)
    valid_mask_test = np.any(X_test != 0, axis=1)

    median_feats = np.median(X_train[valid_mask_train], axis=0)
    X_train[~valid_mask_train] = median_feats
    X_test[~valid_mask_test] = median_feats

    knn = KNeighborsClassifier(
            n_neighbors=3, 
            weights='uniform', 
            metric='euclidean'
            )
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    # Evaluate
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=cfg['names']))

    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

if __name__ == "__main__":
    main()
