import cv2
import dlib
from scipy.spatial import distance
import pygame
import time

# Initialize Pygame for audio alerts
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('alert.wav')

# Function to calculate EAR
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize face detector and facial landmarks predictor using Dlib
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define EAR thresholds and duration
drowsy_threshold = 0.26   # Lower threshold for detecting drowsiness
alert_threshold = 0.24  # Higher threshold for confirming alertness
drowsy_duration = 15      # Number of consecutive frames below drowsy_threshold to trigger drowsiness

# Control frame processing rate (adjust as needed)
frame_processing_interval = 1.0 / 30  # 30 frames per second

consecutive_drowsy_frames = 0

while True:
    start_time = time.time()  # Record start time of frame processing
    
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = hog_face_detector(gray)
    
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        # Extract eye coordinates and draw lines
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        
        # Check for drowsiness and play alert sound
        if EAR < drowsy_threshold:
            consecutive_drowsy_frames += 1
            if consecutive_drowsy_frames >= drowsy_duration:
                cv2.putText(frame, "DROWSY", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                cv2.putText(frame, "Are you Sleepy?", (20, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                print("Drowsy")
                # Play alert sound
                alert_sound.play()
        else:
            consecutive_drowsy_frames = 0  # Reset consecutive drowsy frames

        # Check for alertness and reset drowsiness state
        if EAR > alert_threshold:
            print("Alert")
            consecutive_drowsy_frames = 0  # Reset consecutive drowsy frames

        print(EAR)

    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)

    # Check if 'q' key is pressed to exit the loop
    if key == ord('q'):
        break

    # Calculate time elapsed for frame processing and add delay if needed
    elapsed_time = time.time() - start_time
    if elapsed_time < frame_processing_interval:
        time.sleep(frame_processing_interval - elapsed_time)

cap.release()
cv2.destroyAllWindows()
