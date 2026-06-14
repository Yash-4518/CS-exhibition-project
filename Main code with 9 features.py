import cv2 #for webcam and image handling
import mediapipe as mp #mediapipe for face landmarks
import time
import numpy as np
import pygame #to play alert sound
from tensorflow.keras.models import load_model #trained model ko load karne ke liye
from scipy.spatial import distance
import joblib

#%% ---------------- Setup ----------------
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('alert.wav')    #to load alert sound

# Load trained ML model
model = load_model("/Users/yashsharma/Desktop/Drowsiness/drowsiness_model.h5")  

# load scaler (agar training script ne scaler save kia tha)
scaler = None
try:
    scaler = joblib.load("/Users/yashsharma/Desktop/Drowsiness/scaler.pkl")
except Exception as e:
    print("Scaler not found or failed to load. Continuing without scaler. Error:", e)
    scaler = None

# Initialize video capture
cap = cv2.VideoCapture(0)  #webcam on karre h

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Control frame processing rate (adjust as needed)
frame_processing_interval = 1.0 / 30  # 30 frames per second

# Counters for smooth transition (buffer zone)
consecutive_drowsy_frames = 0
consecutive_alert_frames = 0
required_frames = 15   # Smooth transition buffer (0.5 sec if fps ~30)

# Thresholds for MAR
MAR_threshold = 0.67
MAR_duration = 17
consecutive_MAR_frames = 0

#%% ---------------- Helper Functions ----------------
# Function to calculate EAR
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    # avoiding division by zero
    if C == 0:
        return 0.0
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

# Distance function
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Extract all 9 features from normalized landmarks (MUST match training script)
def extract_features(mp_points_norm):
    try:
        # left eye: [33, 160, 158, 133, 153, 144]
        # right eye: [362, 385, 387, 263, 373, 380]
        leftEye_norm  = [mp_points_norm[i] for i in (33, 160, 158, 133, 153, 144)]
        rightEye_norm = [mp_points_norm[i] for i in (362, 385, 387, 263, 373, 380)]

        left_EAR  = calculate_EAR(leftEye_norm)
        right_EAR = calculate_EAR(rightEye_norm)
        avg_EAR   = (left_EAR + right_EAR) / 2.0
        EAR_diff  = abs(left_EAR - right_EAR)  # asymmetry between eyes

        # -- Mouth indices for MAR (MediaPipe) --
        # Using approximate inner-lip indices to match training
        p60_norm = mp_points_norm[78]   # left-ish corner
        p64_norm = mp_points_norm[308]  # right-ish corner
        p61_norm = mp_points_norm[81]
        p67_norm = mp_points_norm[311]
        p62_norm = mp_points_norm[13]
        p66_norm = mp_points_norm[14]
        p63_norm = mp_points_norm[82]
        p65_norm = mp_points_norm[312]

        # Compute MAR using normalized coords (to match training)
        v1    = euclidean_dist(p61_norm, p67_norm)
        v2    = euclidean_dist(p62_norm, p66_norm)
        v3    = euclidean_dist(p63_norm, p65_norm)
        width = euclidean_dist(p60_norm, p64_norm)
        MAR   = (v1 + v2 + v3) / (2.0 * width) if width != 0 else 0.0

        # -- Face bounding box landmarks --
        face_top    = mp_points_norm[10]   # forehead top
        face_bottom = mp_points_norm[152]  # chin
        face_left   = mp_points_norm[234]  # left cheek
        face_right  = mp_points_norm[454]  # right cheek
        face_height = euclidean_dist(face_top, face_bottom)
        face_width  = euclidean_dist(face_left, face_right)
        face_AR     = face_width / face_height if face_height != 0 else 0.0  # face aspect ratio

        # mouth open ratio normalized by face height
        mouth_open       = euclidean_dist(p62_norm, p66_norm)
        mouth_open_ratio = mouth_open / face_height if face_height != 0 else 0.0

        # avg eye height normalized by face height
        left_eye_h     = euclidean_dist(leftEye_norm[1],  leftEye_norm[5])
        right_eye_h    = euclidean_dist(rightEye_norm[1], rightEye_norm[5])
        eye_area_ratio = ((left_eye_h + right_eye_h) / 2.0) / face_height if face_height != 0 else 0.0

        # head tilt: angle of line connecting eye corners
        lx, ly = mp_points_norm[33]
        rx, ry = mp_points_norm[263]
        head_tilt = abs(ry - ly) / (abs(rx - lx) + 1e-6)

    except IndexError:
        # safety fallback
        return None, 0.0, 0.0

    # build feature array shape (1, 9) — same order as training script
    features = np.array([[
        left_EAR, right_EAR, avg_EAR, EAR_diff,
        MAR, mouth_open_ratio, eye_area_ratio,
        head_tilt, face_AR
    ]], dtype=np.float32)

    return features, round(avg_EAR, 2), round(MAR, 2)

#%% ---------------- Main Loop ----------------
while True:
    start_time = time.time()  # Record start time of frame processing
    
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe expects RGB frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # default values
    EAR = 0.0
    MAR = 0.0

    if results.multi_face_landmarks:
        # take first face only (max_num_faces=1 above)
        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        # build two lists:
        # mp_points_px -> pixel coords for drawing
        # mp_points_norm -> normalized coords (0..1) for feature computation (to match training)
        mp_points_px = []
        mp_points_norm = []
        for lm in face_landmarks.landmark:
            mp_x = int(lm.x * w)
            mp_y = int(lm.y * h)
            mp_points_px.append((mp_x, mp_y))
            mp_points_norm.append((lm.x, lm.y))

        # -- Build leftEye and rightEye lists using MediaPipe indices --
        # left eye: [33, 160, 158, 133, 153, 144]
        # right eye: [362, 385, 387, 263, 373, 380]
        try:
            leftEye_px = [mp_points_px[i] for i in (33, 160, 158, 133, 153, 144)]
            rightEye_px = [mp_points_px[i] for i in (362, 385, 387, 263, 373, 380)]
        except IndexError:
            # safety fallback
            leftEye_px = [(0,0)]*6
            rightEye_px = [(0,0)]*6

        # draw eye contours (green) - keeping your comment style
        for i in range(len(leftEye_px)):
            x, y = leftEye_px[i]
            x2, y2 = leftEye_px[(i+1) % len(leftEye_px)]
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1) #greeen line draw karra h
        for i in range(len(rightEye_px)):
            x, y = rightEye_px[i]
            x2, y2 = rightEye_px[(i+1) % len(rightEye_px)]
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)  # greeen line draw karra h

        # pixel points for drawing mouth lines
        try:
            p60_px = mp_points_px[78]
            p64_px = mp_points_px[308]
            p61_px = mp_points_px[81]
            p67_px = mp_points_px[311]
            p62_px = mp_points_px[13]
            p66_px = mp_points_px[14]
            p63_px = mp_points_px[82]
            p65_px = mp_points_px[312]
        except IndexError:
            # safety fallback
            p60_px = p64_px = p61_px = p62_px = p63_px = p65_px = p66_px = p67_px = (0,0)

        # Draw mouth lines similar to before (blue verticals, green horizontal)
        cv2.line(frame, p61_px, p67_px, (255, 0, 0), 2)
        cv2.line(frame, p62_px, p66_px, (255, 0, 0), 2)
        cv2.line(frame, p63_px, p65_px, (255, 0, 0), 2)
        cv2.line(frame, p60_px, p64_px, (0, 255, 0), 2)

        # ---------- Extract all 9 features (to match training script) ----------
        features_raw, EAR, MAR = extract_features(mp_points_norm)

        # Display EAR & MAR value
        cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"MAR: {MAR:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ---------- ML Prediction ----------
        if features_raw is not None:

            # scale features if scaler available (recommended)
            if scaler is not None:
                try:
                    features_scaled = scaler.transform(features_raw)
                except Exception as e:
                    features_scaled = features_raw
                    print("Scaler transform failed, using raw features. Error:", e)
            else:
                features_scaled = features_raw

            # model expects scaled features if trained that way
            try:
                prediction = model.predict(features_scaled, verbose=0)[0][0]  # sigmoid output (0–1)
            except Exception as e:
                # fallback: if model expects raw features, try raw
                prediction = model.predict(features_raw, verbose=0)[0][0]

            if prediction > 0.5:  # Drowsy
                consecutive_drowsy_frames += 1  #counter badhana
                consecutive_alert_frames = 0
                if consecutive_drowsy_frames >= required_frames:  # agr required frames ke liye hit hua to drowsy
                    cv2.putText(frame, "DROWSY", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                    cv2.putText(frame, "Are you Sleepy?", (20, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    print("Drowsy")
                    alert_sound.play()   # Play alert sound
            else:  # Alert
                consecutive_alert_frames += 1
                consecutive_drowsy_frames = 0
                if consecutive_alert_frames >= required_frames:
                    cv2.putText(frame, "ALERT", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
                    print("Alert")

        # ---------- Yawning Detection ----------
        if MAR > MAR_threshold:
            consecutive_MAR_frames += 1
            if consecutive_MAR_frames >= MAR_duration:
                cv2.putText(frame, "Yawning Detected!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
        else:
            consecutive_MAR_frames = 0

        print(EAR, MAR)

    cv2.imshow("Drowsiness + Yawn Detection (ML + Smooth)", frame)

    key = cv2.waitKey(1)  # means that code will wait for 1 millisecond ..if q is pressed then break else continue
    if key == ord('q'):
        break

    # Calculate time elapsed for frame processing and add delay if needed
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_processing_interval:
        time.sleep(frame_processing_interval - elapsed_time)

cap.release()
cv2.destroyAllWindows()
