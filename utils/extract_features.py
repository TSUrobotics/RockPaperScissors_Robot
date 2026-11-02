import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize this ONCE outside your training loop (e.g., in main())
def init_landmark_model(model_path: str):
    """Loads the GestureRecognizer model using the modern tasks API."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    return vision.GestureRecognizer.create_from_options(options)

def extract_landmark_features(model, batch_imgs, batch_labels):
    """
    Extracts 63-dim hand landmark features (21 pts × 3 coords) from a batch.
    """
    X_feats, y_feats = [], []
    
    for img_t, lbl in zip(batch_imgs, batch_labels):
        # 1. Convert to numpy
        img_np = img_t.cpu().numpy() if hasattr(img_t, 'cpu') else np.array(img_t)
        
        # 2. Permute from (C, H, W) -> (H, W, C) if needed
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
            
        # 3. Denormalize: [-1, 1] -> [0, 255]
        img_np = np.clip((img_np * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        
        # 4. MediaPipe tasks expects BGR format
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 5. Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_bgr)
        
        # 6. Run recognition
        result = model.recognize(mp_image)
        
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            feats = [landmarks[i].x for i in range(21)] + \
                    [landmarks[i].y for i in range(21)] + \
                    [landmarks[i].z for i in range(21)]
            X_feats.append(feats)
        else:
            X_feats.append(np.zeros(63))
            
        y_feats.append(lbl)
        
    return np.array(X_feats, dtype=np.float32), np.array(y_feats, dtype=np.int64)
