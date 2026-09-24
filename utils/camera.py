import cv2
import torch
from PIL import Image

def is_raspberry_pi():
    try:
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            return 'raspberry pi' in f.read().lower()
    except:
        return False

# Camera inference loop
def segment(image, background, threshold=15):
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

def run_camera(model, class_names, device, transform,
               cam_index: int = 3,
               width: int = 640,
               height: int = 480) -> None:
    # ROI Coordinates (top, right, bottom, left)
    top, right, bottom, left = 10, 350, 225, 590

    # Normally, 0 for built-in webcam and 1&2 for external USB devices
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("Press 's' to start, 'q' to quit.")

    raspberry_pi = False
    if is_raspberry_pi():
        from utils.init_hand import create_hand
        hand, gestures = create_hand()
        raspberry_pi = True


    last_pred = ""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # Clone the frame so we can draw the box without affecting processing
        clone = frame.copy()
    
        keypress = cv2.waitKey(1) & 0xFF

        with torch.no_grad():
            results = model.predict(
                    source=frame, # We need the direct frame.
                    device=device,
                    )
            res = results[0]  # Get the first result from the list

            pred_id = -1
            confidence = 0.0

            # Extract prediction based on model type
            if res.probs is not None:
                # For Classification models (YOLOv8-cls)
                # .probs is a Top1 object containing the index and confidence
                #print(res.probs)
                if res.probs is None:
                    pred_id = -1
                    confidence = 0
                else:
                    pred_id = int(res.probs.top1)
                    confidence = float(res.probs.top1conf)
            elif res.boxes is not None and len(res.boxes) > 0:
                # For Detection models (YOLOv8-detect)
                # Use the first detected box's class
                pred_id = int(res.boxes.cls[0].item())
                confidence = float(res.boxes.conf[0].item())
            else:
                # No detection/classification found
                pred_id = -1
                confidence = 0.0
            
            # Safely get class name
            #pred_name = class_names[pred_id] if pred_id != -1 and pred_id < len(class_names) else "Unknown"
            pred_name = class_names[pred_id]

        # Draw the prediction text (moved inside the if block from original code)
        cv2.putText(clone,
                    f'{pred_name}: {confidence:.2f}',
                    (right + 10, top + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)


        if pred_name != last_pred:
            print(f'Prediction: {pred_name}')
            last_pred = pred_name
        if raspberry_pi:
            # Game Logic
            match pred_name:
                case 'Rock':
                    hand.paper()
                case 'Paper':
                    hand.scissors()
                case 'Scissors':
                    hand.rock()

        cv2.imshow('Prediction', res.plot())

        # Draw the bounding box on the main display
        #cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow('Video Feed', clone)

        if keypress == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if raspberry_pi:
        hand.stop()
