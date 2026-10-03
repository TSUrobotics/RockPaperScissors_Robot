from utils.arguments import set_args
from utils.camera import run_camera
from utils.get_paths import load_yaml_config

import argparse
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

def main():
    args = set_args()

    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.xpu.is_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')

    cfg = load_yaml_config(args.data)

    print(cfg)

    cfg_names = list(cfg['names'])

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model = YOLO()

    model.load(args.ckpt)

    metrics = model.val(data=args.data)

    print(metrics.confusion_matrix.matrix)

    if args.camera:
        run_camera(model, cfg['names'], device, transform, args.camera, args.width, args.height)


if __name__ == "__main__":
    main()
