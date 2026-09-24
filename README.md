# Rock, Paper, Scissors Gesture Recognition

## 1. Automatic Frame Extraction from Video File

To automatically extract frames from a video file, run the script:

```bash
extract_frames.sh $MOVIE
```

where `$MOVIE` is the video file.

## 2. Gesture Labeling

To detect and label the gestures automatically, run the following program:

> [!NOTE]
> Be sure to install the required Python modules from `requirements.txt`, either in a `venv` or globally, before running the following programs.
>
> You will also need `gesture_recognizer.task` from [this link](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer#models).
>
> You can download it with `wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task`.

```bash
python labeler_mediapipe.py --src $FRAMES --out $LABELS
```

where `$FRAMES` is the directory holding the video frames and `$LABELS` is the directory that will hold the corresponding labels.

The labels are written in YOLO format.

## 3. Dataset Split

To randomly split the dataset into training, testing, and valid sets, run the following program.

```bash
python split.py --src_images $FRAMES --src_labels $LABELS --split $SPLIT --out $IMAGES
```

where `$FRAMES` is the directory holding the video frames, `$LABELS` is the directory holding the corresponding labels, `$SPLIT` is a float inclusively between $0$ and $1$, `$IMAGES` is the directory where the selected training, testing, and valid images and labels will be copied to.

The directory structure is in YOLO format.

## 4. Model Training

To train the YOLO model, run the following program:

```bash
python yolo_trainer.py --data $DATA --ckpt $CKPT
```

where `$DATA` is the YOLO yaml file describing the dataset and `$CKPT` is the path to the downloaded pretrained YOLO model size (i.e. `yolo26n.pt`, `yolo26m.pt`, `yolo26l.pt`, etc).

## 5. Model Checking

To check the YOLO model results, run the follwing program:

```bash
python yolo_checker.py --data $DATA --ckpt $CKPT --camera $CAMERA --width $WIDTH --height $HEIGHT
```

where `$DATA` is the YOLO yaml file describing the dataset, `$CKPT` is the path to the trained YOLO model weight, `$CAMERA` an integer that is the index of the video device to be used, `$WIDTH` is an integer that is the width of the camera feed in pixels, `$HEIGHT` is an integer that is the height of the camera feed in pixels, 
