import os
import yaml

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_paths(config, yaml_path, image_type="train"):
    config = load_yaml_config(yaml_path)
    img_dir = os.path.join(os.path.dirname(yaml_path), config[image_type])
    # Replace the last folder name (images) with "labels"
    label_dir = os.path.join(
        os.path.dirname(yaml_path),
        config[image_type].rsplit(os.sep, 1)[0],
        "labels"
    )

    # Bodge
    n = img_dir.rfind('../')
    img_dir = img_dir[:n] + img_dir[n+3:]
    n = label_dir.rfind('../')
    label_dir = label_dir[:n] + label_dir[n+3:]

    print("Image folder :", img_dir)
    print("Labels folder:", label_dir)

    return img_dir, label_dir
