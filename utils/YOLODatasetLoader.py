import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class YOLODatasetLoader(Dataset):
    def __init__(self, img_dir: str, label_dir: str,
                 transforms=None, num_classes: int = 3):
        self.img_dir   = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.num_classes = num_classes

        self.samples = []
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            txt_path = os.path.join(label_dir,
                                    os.path.splitext(img_name)[0] + '.txt')
            if not os.path.isfile(txt_path):
                continue

            with open(txt_path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    continue
                try:
                    cid = int(line.split()[0])
                except ValueError:
                    continue

                if 0 <= cid < self.num_classes:
                    self.samples.append(
                        (os.path.join(img_dir, img_name), cid)
                    )
                else:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]

        # Image
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)

        # Label
        return img, torch.tensor(class_id, dtype=torch.long)
