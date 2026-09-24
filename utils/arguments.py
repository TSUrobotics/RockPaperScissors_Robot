import argparse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        help="Input folder of images."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output folder of labels."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml"
    )
    parser.add_argument(
        "--ckpt", 
        type=str, 
        default="runs/detect/train-6/weights/best.pt",
        help="YOLO model weights checkpoint."
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Video device to chose (Recommended: 3 → indices `ls /dev/video*`).",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Width of video frame (Recommended: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Height of video frame (Recommended: 480).",
    )
    
    return parser.parse_args()
