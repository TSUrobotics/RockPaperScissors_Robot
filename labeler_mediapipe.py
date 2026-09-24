"""
This program was adapted from the mediapipe example notebook from the following link:

https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb
"""
import argparse
import os
import sys
from pathlib import Path
from types import NoneType
import numpy as np
from matplotlib import pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    return parser.parse_args()

def display(image, top_gesture, hand_landmarks, bounding_boxes, fig):
    """Displays one image with its predicted gesture category, score, hand landmarks, and bounding box(es)."""
    image = image.numpy_view()

    if top_gesture is None:
        title = "None (nan)"
    else:
        title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"

    annotated_image = image.copy()
    for landmarks in hand_landmarks:
        mp.tasks.vision.drawing_utils.draw_landmarks(
            annotated_image,
            landmarks,
            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
            mp.tasks.vision.drawing_styles.get_default_hand_landmarks_style(),
            mp.tasks.vision.drawing_styles.get_default_hand_connections_style())

    for box in bounding_boxes:
        cv2.rectangle(
            annotated_image,
            box["top_left"],
            box["bottom_right"],
            color=(255, 0, 0),
            thickness=2
        )

    fig.clear()
    plt.imshow(annotated_image)
    if len(title) > 0:
        plt.title(title, fontsize=16, color='black',
                   fontdict={'verticalalignment': 'center'}, pad=10)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0005)

def get_bounding_boxes(hand_landmarks_list, image_width, image_height, margin=20):
    """
    Computes a pixel-coordinate bounding box for each detected hand.

    Returns a list (one entry per hand) of dicts with the four corners,
    ordered top_left -> top_right -> bottom_right -> bottom_left.
    """
    boxes = []
    for landmarks in hand_landmarks_list:
        xs = [lm.x * image_width for lm in landmarks]
        ys = [lm.y * image_height for lm in landmarks]

        x_min = max(int(min(xs)) - margin, 0)
        x_max = min(int(max(xs)) + margin, image_width - 1)
        y_min = max(int(min(ys)) - margin, 0)
        y_max = min(int(max(ys)) + margin, image_height - 1)

        corners = {
            "top_left": (x_min, y_min),
            "top_right": (x_max, y_min),
            "bottom_right": (x_max, y_max),
            "bottom_left": (x_min, y_max),
        }
        boxes.append(corners)
    return boxes

def bbox_to_yolo(box, image_width, image_height):
    """Converts a pixel-coordinate bounding box (corners) to YOLO format:
    normalized (x_center, y_center, width, height)."""
    x_min, y_min = box["top_left"]
    x_max, y_max = box["bottom_right"]

    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return x_center, y_center, width, height

def gesture_mapping(category_name):
    """
    Map finger count → gesture class id.
    Returns:
        0 → rock   (0 fingers)
        1 → paper  (5 fingers)
        2 → scissors (2 fingers)
        -1 → unknown / transition
    """
    if category_name == "Closed_Fist":
        return 0            # rock
    if category_name == "Open_Palm":
        return 1            # paper
    if category_name == "Victory":
        return 2            # scissors
    return -1               # ambiguous / moving hand

def main():
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'xtick.labelbottom': False,
        'xtick.bottom': False,
        'ytick.labelleft': False,
        'ytick.left': False,
        'xtick.labeltop': False,
        'xtick.top': False,
        'ytick.labelright': False,
        'ytick.right': False
    })

    args = set_args()

    current_dir = os.getcwd()
    model_path = f'{current_dir}/gesture_recognizer.task'
    image_path = Path(args.src).expanduser().resolve()

    IMAGE_FILENAMES = sorted([p for p in image_path.iterdir() if p.suffix.lower() in {".png"}])

    # Preview the images.

    # STEP 1: Import the necessary modules.
    # At top of file where they should be.

    # STEP 2: Create an GestureRecognizer object.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    labels_dir = image_path.parent / image_path.name.replace("frames", "labels")
    labels_dir.mkdir(parents=True, exist_ok=True)

    plt.ion()
    fig = plt.figure(figsize=(6,6))

    all_results = []
    for image_file_name in IMAGE_FILENAMES:
        # STEP 3: Load the input image.
        image = mp.Image.create_from_file(str(image_file_name))

        # STEP 4: Recognize gestures in the input image.
        recognition_result = recognizer.recognize(image)

        # STEP 5: Process the result. In this case, visualize it.
        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
        else:
            top_gesture = None

        hand_landmarks = recognition_result.hand_landmarks or []

        image_height, image_width = image.numpy_view().shape[:2]
        bounding_boxes = get_bounding_boxes(hand_landmarks, image_width, image_height)

        mapped_gesture = gesture_mapping(top_gesture.category_name) if top_gesture else -1
        print(f"{mapped_gesture}\t, {bounding_boxes}\n")

        # Write YOLO-format label file (skip if gesture is ambiguous/unknown).
        label_file_path = labels_dir / f"{image_file_name.stem}.txt"
        with open(label_file_path, "w") as f:
            if mapped_gesture != -1:
                for box in bounding_boxes:
                    x_center, y_center, width, height = bbox_to_yolo(box, image_width, image_height)
                    f.write(f"{mapped_gesture} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        all_results.append({
            "image": image_file_name.name,
            "gesture": mapped_gesture,
            "bounding_boxes": bounding_boxes
        })

        display(image, top_gesture, hand_landmarks, bounding_boxes, fig)

    plt.ioff()
    plt.show()

    # Save all bounding box coordinates to disk.
    if args.out_dir:
        import json
        out_path = Path(args.out_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "bounding_boxes.json", "w") as f:
            json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
