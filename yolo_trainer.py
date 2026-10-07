from utils.arguments import set_args
from utils.get_torch_device import torch_device

from ultralytics import YOLO

def main():
    args = set_args()

    device = torch_device()

    model = YOLO(args.ckpt)

    model.train(data=args.data, epochs=300, device=device)

if __name__ == "__main__":
    main()
