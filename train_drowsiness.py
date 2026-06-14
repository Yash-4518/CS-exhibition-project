import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DROWSY_DIR = "/Users/yashsharma/Desktop/Drowsiness/Driver Drowsiness Dataset (DDD)/Drowsy"
ALERT_DIR  = "/Users/yashsharma/Desktop/Drowsiness/Driver Drowsiness Dataset (DDD)/Non Drowsy"
MODEL_OUT  = "/Users/yashsharma/Desktop/Drowsiness/drowsiness_model.h5"
SCALER_OUT = "/Users/yashsharma/Desktop/Drowsiness/scaler.pkl"

# ─────────────────────────────────────────────
#  MediaPipe setup
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────────
#  Feature helpers
# ─────────────────────────────────────────────
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_features_from_frame(frame):
    """
    Returns 9 features:
      0  left_EAR
      1  right_EAR
      2  avg_EAR
      3  EAR_diff          (asymmetry between eyes)
      4  MAR
      5  mouth_open_ratio  (vertical / face height)
      6  eye_area_ratio    (avg eye height / face height)
      7  head_tilt         (horizontal eye line angle)
      8  face_aspect_ratio (face width / face height)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    lms = results.multi_face_landmarks[0].landmark
    pts = [(lm.x, lm.y) for lm in lms]

    try:
        # ── Eyes ──
        left_eye  = [pts[i] for i in (33, 160, 158, 133, 153, 144)]
        right_eye = [pts[i] for i in (362, 385, 387, 263, 373, 380)]
        left_EAR  = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)
        avg_EAR   = (left_EAR + right_EAR) / 2.0
        EAR_diff  = abs(left_EAR - right_EAR)

        # ── Mouth ──
        p60 = pts[78];  p64 = pts[308]
        p61 = pts[81];  p67 = pts[311]
        p62 = pts[13];  p66 = pts[14]
        p63 = pts[82];  p65 = pts[312]
        v1    = euclidean_dist(p61, p67)
        v2    = euclidean_dist(p62, p66)
        v3    = euclidean_dist(p63, p65)
        width = euclidean_dist(p60, p64)
        MAR   = (v1 + v2 + v3) / (2.0 * width) if width != 0 else 0.0

        # ── Face bounding box ──
        face_top    = pts[10]   # forehead top
        face_bottom = pts[152]  # chin
        face_left   = pts[234]  # left cheek
        face_right  = pts[454]  # right cheek
        face_height = euclidean_dist(face_top, face_bottom)
        face_width  = euclidean_dist(face_left, face_right)
        face_AR     = face_width / face_height if face_height != 0 else 0.0

        # ── Mouth open ratio (normalized by face height) ──
        mouth_open = euclidean_dist(p62, p66)
        mouth_open_ratio = mouth_open / face_height if face_height != 0 else 0.0

        # ── Eye area ratio (avg eye vertical span / face height) ──
        left_eye_h  = euclidean_dist(left_eye[1],  left_eye[5])
        right_eye_h = euclidean_dist(right_eye[1], right_eye[5])
        eye_area_ratio = ((left_eye_h + right_eye_h) / 2.0) / face_height if face_height != 0 else 0.0

        # ── Head tilt (angle of line connecting eye corners) ──
        lx, ly = pts[33]
        rx, ry = pts[263]
        head_tilt = abs(ry - ly) / (abs(rx - lx) + 1e-6)

    except IndexError:
        return None

    return [
        round(left_EAR,         4),
        round(right_EAR,        4),
        round(avg_EAR,          4),
        round(EAR_diff,         4),
        round(MAR,              4),
        round(mouth_open_ratio, 4),
        round(eye_area_ratio,   4),
        round(head_tilt,        4),
        round(face_AR,          4),
    ]

# ─────────────────────────────────────────────
#  Extract from folder
# ─────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def extract_from_folder(folder_path, label):
    X, y = [], []
    files = os.listdir(folder_path)
    total = len(files)
    print(f"\nProcessing '{folder_path}' ({total} files, label={label}) ...")

    for idx, fname in enumerate(files):
        fpath = os.path.join(folder_path, fname)
        ext   = os.path.splitext(fname)[1].lower()

        if ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(fpath)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 3 != 0:
                    continue
                feats = extract_features_from_frame(frame)
                if feats is not None:
                    X.append(feats)
                    y.append(label)
            cap.release()

        elif ext in IMAGE_EXTS:
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            feats = extract_features_from_frame(frame)
            if feats is not None:
                X.append(feats)
                y.append(label)

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"  [{idx+1}/{total}] samples so far: {len(X)}")

    return X, y

# ─────────────────────────────────────────────
#  Collect data
# ─────────────────────────────────────────────
X_drowsy, y_drowsy = extract_from_folder(DROWSY_DIR, label=1)
X_alert,  y_alert  = extract_from_folder(ALERT_DIR,  label=0)

X = np.array(X_drowsy + X_alert, dtype=np.float32)
y = np.array(y_drowsy + y_alert, dtype=np.float32)

print(f"\nTotal samples  : {len(X)}")
print(f"  Drowsy (1)   : {int(y.sum())}")
print(f"  Alert  (0)   : {int((1 - y).sum())}")
print(f"  Features     : {X.shape[1]}  (was 2 before, now {X.shape[1]})")

if len(X) == 0:
    raise ValueError("No features extracted. Check folder paths.")

# ─────────────────────────────────────────────
#  Scale
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_OUT)
print(f"Scaler saved → {SCALER_OUT}")

# ─────────────────────────────────────────────
#  Split
# ─────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

# ─────────────────────────────────────────────
#  Model  (deeper, with BatchNorm)
# ─────────────────────────────────────────────
n_features = X.shape[1]

model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dense(1,  activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ─────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ─────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────
model.save(MODEL_OUT)
print(f"\nModel saved → {MODEL_OUT}")

# ─────────────────────────────────────────────
#  Plot
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],     label='Train acc')
ax1.plot(history.history['val_accuracy'], label='Val acc')
ax1.set_title('Accuracy'); ax1.legend()
ax2.plot(history.history['loss'],     label='Train loss')
ax2.plot(history.history['val_loss'], label='Val loss')
ax2.set_title('Loss'); ax2.legend()
plt.tight_layout()
plt.savefig("/Users/yashsharma/Desktop/Drowsiness/training_curves.png")
print("Training curves saved → training_curves.png")
plt.show()

# ─────────────────────────────────────────────
#  Evaluate
# ─────────────────────────────────────────────
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation accuracy : {val_acc*100:.2f}%")
print(f"Validation loss     : {val_loss:.4f}")
