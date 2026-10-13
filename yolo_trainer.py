import sys
from datetime import datetime

from utils.arguments import set_args
from utils.get_torch_device import torch_device

from ultralytics import YOLO

def main():
    args = set_args()
    date = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")

    model = YOLO(args.ckpt)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        device=torch_device(),
        name=f"{args.ckpt[-10:-3]}-{args.data[0:26]}-{date}",
        pretrained=True
    )

if __name__ == "__main__":
    main()
