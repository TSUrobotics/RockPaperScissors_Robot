from utils.arguments import set_args
from utils.get_torch_device import torch_device

import pprint
from ultralytics import YOLO

def main():
    args = set_args()

    device = torch_device()

    model = YOLO()

    model.load(args.ckpt)

    results = model.val(data=args.data, plots=True)

    print(f"mAP50: {results.box.map50}")
    print(f"Precision: {results.box.p}")
    print(f"Recall: {results.box.r}")
    print(f"F1-Score: {results.box.f1}")

    pprint.pprint(results.summary())

if __name__ == "__main__":
    main()
