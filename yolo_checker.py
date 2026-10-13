from utils.arguments import set_args
from utils.camera import run_camera
from utils.get_paths import load_yaml_config
from utils.get_torch_device import torch_device

import argparse
from ultralytics import YOLO

def main():
    args = set_args()

    device = torch_device()

    model = YOLO(args.ckpt)

    if args.camera:
        run_camera(model, args.camera, args.width, args.height)
    else:
        print("Please give camera index, width, and height.")


if __name__ == "__main__":
    main()
