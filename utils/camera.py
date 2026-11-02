import time

import cv2
import numpy as np
import torch
from PIL import Image

def is_raspberry_pi():
    try:
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            return 'raspberry pi' in f.read().lower()
    except:
        return False

# Camera inference loop
def segment(image, background, threshold=25):
    # Compute absolute difference between background and current frame
    diff = cv2.absdiff(background, image)
    # Threshold to get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    # Find contours
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None
    else:
        # Return the largest contour (the hand)
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def run_camera(model,
               cam_index: int = 3,
               width: int = 640,
               height: int = 480) -> None:
    
    PREDICT_EVERY = 10  # Run inference every N frames

    YOLO_CONF_THRESHOLD = 0.4
    YOLO_IOU_THRESHOLD = 0.45

    # Open webcam
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # FPS tracking
    process_fps = time.time()
    frame_count = 0
    display_fps = 0.0
    
    # Store last prediction result to persist across frames
    last_result = None

    print("🎥 Press 'q' or 'ESC' to quit | YOLO Live Detection")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ Failed to grab frame. Exiting.")
            break

        # Mirror view for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Convert BGR → RGB (YOLO expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO Detection (every N frames)
        frame_count += 1
        if frame_count % PREDICT_EVERY == 0:
            results = model(rgb_frame, conf=YOLO_CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD, verbose=False)
            last_result = results[0]  # Store for persistence

        # Draw stored result on every frame
        if last_result is not None:
            result = last_result

            # Draw bounding boxes
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {conf:.2f}"

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Segmentation Masks
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks
                colors = result.colors if hasattr(result, 'colors') else None

                for i, mask in enumerate(masks.xy):
                    mask_poly = np.array(mask, dtype=np.int32)
                    # Pick a color for this mask
                    if colors is not None and i < len(colors):
                        # Convert RGB (from Ultralytics) → BGR (for OpenCV)
                        color = tuple(int(c) for c in reversed(colors[i % len(colors)]))
                    else:
                        color = (int(np.random.randint(0, 255)),
                                 int(np.random.randint(0, 255)),
                                 int(np.random.randint(0, 255)))

                    # Draw filled polygon with transparency
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [mask_poly], color)
                    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

                    # Optionally draw mask outline for crisp edges
                    cv2.polylines(frame, [mask_poly], isClosed=True,
                                  color=color, thickness=1)

        # FPS Overlay
        process_fps = time.time() - process_fps
        if frame_count >= 30:
            display_fps = int(frame_count / process_fps)
            frame_count = 0
            process_fps = 0.0

        cv2.putText(frame, f"FPS: {display_fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "YOLO Detection", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display
        cv2.imshow("YOLO Live Detection", frame)

        # Exit on 'q' or 'ESC'
        if cv2.waitKey(5) & 0xFF == 27:  # 27 = ESC
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
