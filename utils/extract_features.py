import numpy as np
import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize this ONCE outside your training loop (e.g., in main())
def init_landmark_model(model_path: str):
    """Loads the GestureRecognizer model using the modern tasks API."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    return vision.GestureRecognizer.create_from_options(options)

def calculate_angle(landmark_a, landmark_b, landmark_c):
    """
    Calculates the angle at landmark_b formed by landmarks a-b-c.
    Implements the formula from the instructions:
    theta = arccos( (a . b) / (||a|| ||b||) )
    
    Args:
        landmark_a: Base joint (e.g., MCP)
        landmark_b: Junction joint (e.g., PIP)
        landmark_c: Tip joint (e.g., DIP)
        
    Returns:
        float: Angle in degrees.
    """
    # Vectors from the junction (b) to the endpoints (a and c)
    # Vector a = p_a - p_b
    # Vector b = p_c - p_b
    vec_ab = np.array([landmark_a.x - landmark_b.x, landmark_a.y - landmark_b.y])
    vec_cb = np.array([landmark_c.x - landmark_b.x, landmark_c.y - landmark_b.y])
    
    # Normalize vectors
    norm_ab = np.linalg.norm(vec_ab)
    norm_cb = np.linalg.norm(vec_cb)
    
    # Handle potential zero-length vectors (degenerate cases)
    if norm_ab < 1e-8 or norm_cb < 1e-8:
        return 0.0
        
    vec_ab /= norm_ab
    vec_cb /= norm_cb
    
    # Dot product
    dot_product = np.dot(vec_ab, vec_cb)
    
    # Clamp to avoid numerical errors outside [-1, 1] which would cause NaN in acos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    angle = math.degrees(math.acos(dot_product))
    return angle

def extract_rps_features(model, batch_imgs, batch_labels):
    """Extract the 10‑dimensional RPS feature vector for a batch of images.

    The feature vector follows the ChatGPT instructions:

    * **Five finger‑bending angles** (thumb, index, middle, ring, little) –
      computed as the interior angle at the PIP/DIP joint using three consecutive
      landmarks.
    * **Five normalized fingertip‑to‑palm distances** – the Euclidean distance
      from each fingertip to the middle‑finger MCP (landmark 9), divided by the
      palm width ``s`` (distance between landmarks 5 and 17).

    The function works with the modern MediaPipe *tasks* API (``vision.GestureRecognizer``).
    It returns a ``(N, 10)`` NumPy array of ``float32`` features and a matching
    label array.
    """
    X_feats, y_feats = [], []

    for img_t, lbl in zip(batch_imgs, batch_labels):
        # ------------------------------------------------------------
        # 1️⃣ Convert the torch tensor to a NumPy image (H, W, C).
        # ------------------------------------------------------------
        img_np = img_t.cpu().numpy() if hasattr(img_t, "cpu") else np.array(img_t)
        if img_np.ndim == 3 and img_np.shape[0] == 3:  # (C, H, W) → (H, W, C)
            img_np = np.transpose(img_np, (1, 2, 0))
        # Undo the ``transforms.Normalize`` that maps [0, 1] → [-1, 1]
        img_np = np.clip((img_np * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        # MediaPipe expects BGR images
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ------------------------------------------------------------
        # 2️⃣ Run the MediaPipe hand‑landmark recognizer.
        # ------------------------------------------------------------
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_bgr)
        result = model.recognize(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Helper to fetch (x, y) coordinates of a landmark (already normalized).
            def get_coord(idx):
                lm = landmarks[idx]
                return lm.x, lm.y

            # --------------------------------------------------------
            # Palm width ``s`` – distance between index‑MCP (5) and
            # little‑MCP (17).  This is the normalization factor.
            # --------------------------------------------------------
            x5, y5 = get_coord(5)
            x17, y17 = get_coord(17)
            s = math.hypot(x5 - x17, y5 - y17)
            if s < 1e-5:  # Guard against division‑by‑zero for side‑view occlusion
                s = 1e-5

            # --------------------------------------------------------
            # 3️⃣ Finger‑bending angles (in degrees).
            # --------------------------------------------------------
            theta_thumb = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
            theta_index = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
            theta_middle = calculate_angle(landmarks[9], landmarks[10], landmarks[11])
            theta_ring = calculate_angle(landmarks[13], landmarks[14], landmarks[15])
            theta_little = calculate_angle(landmarks[17], landmarks[18], landmarks[19])

            # --------------------------------------------------------
            # 4️⃣ Normalized fingertip distances.
            # --------------------------------------------------------
            x_ref, y_ref = get_coord(9)  # Middle‑finger MCP as reference
            tip_indices = [4, 8, 12, 16, 20]
            fingertip_dists = []
            for idx in tip_indices:
                x_tip, y_tip = get_coord(idx)
                dist = math.hypot(x_tip - x_ref, y_tip - y_ref)
                fingertip_dists.append(dist / s)

            # Combine into a single 10‑D vector.
            features = [theta_thumb, theta_index, theta_middle, theta_ring, theta_little] + fingertip_dists
            X_feats.append(features)
        else:
            # No hand detected – use a zero vector (will be replaced later).
            X_feats.append([0.0] * 10)

        y_feats.append(lbl)

    return np.array(X_feats, dtype=np.float32), np.array(y_feats, dtype=np.int64)
