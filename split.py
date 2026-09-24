import argparse
import sys
from pathlib import Path
import shutil
import random
from tqdm import tqdm

def set_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test for YOLO.")
    parser.add_argument("--src_images", type=str, required=True, help="Path to source images.")
    parser.add_argument("--src_labels", type=str, required=True, help="Path to source labels.")
    parser.add_argument("--split", type=float, default=0.8, help="Fraction for training (default: 0.8)")
    parser.add_argument("--out", type=str, default=None, help="Output base directory.")
    return parser.parse_args()

def copy_pair(stems, img_dest, lbl_dest, image_files, label_files, desc="Copying"):
    for stem in tqdm(stems, desc=desc, unit=" pairs", leave=True):
        shutil.copy2(image_files[stem], img_dest)
        shutil.copy2(label_files[stem], lbl_dest)

def main():
    args = set_args()
    src_images = Path(args.src_images).expanduser().resolve()
    src_labels = Path(args.src_labels).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else src_images.parent / "dataset"

    if not src_images.is_dir() or not src_labels.is_dir():
        sys.exit("Error: Source directories do not exist.")

    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    label_ext = '.txt'
    
    image_files = {p.stem: p for p in src_images.iterdir() if p.is_file() and p.suffix.lower() in image_exts}
    label_files = {p.stem: p for p in src_labels.iterdir() if p.is_file() and p.suffix.lower() == label_ext}
    
    common_stems = sorted(set(image_files.keys()) & set(label_files.keys()))
    if not common_stems:
        sys.exit("Error: No matching image-label pairs found.")

    random.seed(42)
    random.shuffle(common_stems)

    n_total = len(common_stems)
    n_train = int(n_total * args.split)
    n_val = int((n_total - n_train) * 0.5)
    
    train_stems = common_stems[:n_train]
    val_stems = common_stems[n_train:n_train + n_val]
    test_stems = common_stems[n_train + n_val:]

    print(f"Total pairs: {n_total} | Train: {len(train_stems)} | Val: {len(val_stems)} | Test: {len(test_stems)}")

    # Create directories, YOLO requires train and val
    splits = {
        "train": (out_dir / "train" / "images", out_dir / "train" / "labels"),
        "val": (out_dir / "val" / "images", out_dir / "val" / "labels"),
        "test": (out_dir / "test" / "images", out_dir / "test" / "labels")
    }
    for split_name, (img_d, lbl_d) in splits.items():
        img_d.mkdir(parents=True, exist_ok=True)
        lbl_d.mkdir(parents=True, exist_ok=True)

    # Copy files
    copy_pair(train_stems, splits["train"][0], splits["train"][1], image_files, label_files, "Training Split")
    copy_pair(val_stems, splits["val"][0], splits["val"][1], image_files, label_files, "Validation Split")
    copy_pair(test_stems, splits["test"][0], splits["test"][1], image_files, label_files, "Testing Split")

    print(f"\nDataset successfully split into: {out_dir}")

if __name__ == "__main__":
    main()

