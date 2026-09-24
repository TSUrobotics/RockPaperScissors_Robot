from utils.arguments import set_args

import torch
from ultralytics import YOLO

def main():
    args = set_args()

    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO(args.ckpt)

    model.train(data=args.data, epochs=128, device=device)

if __name__ == "__main__":
    main()
