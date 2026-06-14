import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import csv
import os
import time

# ─────────────────────────────────────────────
#  This script records YOUR face via webcam
#  and saves EAR/MAR/features to a CSV file
#  Run it twice:
#    python collect_data.py  → sit normally (ALERT)
#    python collect_data.py  → close eyes/look sleepy (DROWSY)
# ─────────────────────────────────────────────

SAVE_PATH = "/Users/yashsharma/Desktop/Drowsiness/webcam_data.csv"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   refine_landmarks=True, min_detection_confidence=0.5)

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def extract_features(pts):
    le = [pts[i] for i in (33, 160, 158, 133, 153, 144)]
    re = [pts[i] for i in (362, 385, 387, 263, 373, 380)]
    left_EAR  = calculate_EAR(le)
    right_EAR = calculate_EAR(re)
    avg_EAR   = (left_EAR + right_EAR) / 2.0
    EAR_diff  = abs(left_EAR - right_EAR)
    v1  = euclidean_dist(pts[81],  pts[311])
    v2  = euclidean_dist(pts[13],  pts[14])
    v3  = euclidean_dist(pts[82],  pts[312])
    w   = euclidean_dist(pts[78],  pts[308])
    MAR = (v1 + v2 + v3) / (2.0 * w) if w != 0 else 0.0
    fh  = euclidean_dist(pts[10],  pts[152])
    fw  = euclidean_dist(pts[234], pts[454])
    face_AR          = fw / fh if fh != 0 else 0.0
    mouth_open_ratio = euclidean_dist(pts[13], pts[14]) / fh if fh != 0 else 0.0
    eye_area_ratio   = ((euclidean_dist(le[1], le[5]) + euclidean_dist(re[1], re[5])) / 2) / fh if fh != 0 else 0.0
    lx, ly = pts[33]; rx, ry = pts[263]
    head_tilt = abs(ry - ly) / (abs(rx - lx) + 1e-6)
    return [left_EAR, right_EAR, avg_EAR, EAR_diff,
            MAR, mouth_open_ratio, eye_area_ratio, head_tilt, face_AR]

# ── Ask label ──
print("\n==============================")
print("What are you going to record?")
print("  0 = ALERT  (sit normally, eyes open)")
print("  1 = DROWSY (close eyes, look sleepy, nod head)")
label = input("Enter 0 or 1: ").strip()
if label not in ('0', '1'):
    print("Invalid input. Enter 0 or 1.")
    exit()
label = int(label)

label_name = "ALERT" if label == 0 else "DROWSY"
print(f"\nRecording {label_name} data.")
print("Press S to START recording, Q to QUIT and save.\n")

cap = cv2.VideoCapture(0)
recording = False
samples = []

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "Press S to start" if not recording else f"Recording {label_name}... ({len(samples)} samples)"
    color  = (0, 255, 0) if not recording else (0, 0, 255)
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Label: {label_name}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if recording and results.multi_face_landmarks:
        try:
            pts  = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
            feat = extract_features(pts)
            samples.append(feat + [label])
        except Exception as e:
            pass

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1)

    if key == ord('s') or key == ord('S'):
        recording = True
        print("Recording started...")

    if key == ord('q') or key == ord('Q'):
        print(f"Stopping. Collected {len(samples)} samples.")
        break

cap.release()
cv2.destroyAllWindows()

# ── Save to CSV ──
if len(samples) == 0:
    print("No samples collected. Exiting.")
    exit()

file_exists = os.path.exists(SAVE_PATH)
with open(SAVE_PATH, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['left_EAR','right_EAR','avg_EAR','EAR_diff',
                         'MAR','mouth_open_ratio','eye_area_ratio',
                         'head_tilt','face_AR','label'])
    writer.writerows(samples)

print(f"\nSaved {len(samples)} {label_name} samples to {SAVE_PATH}")
print("Run again with label 1 if you only did label 0, or vice versa.")
