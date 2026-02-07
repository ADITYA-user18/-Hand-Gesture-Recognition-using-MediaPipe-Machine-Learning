import os
import pickle                    # For saving extracted features
import mediapipe as mp           # For hand landmark detection
import cv2                       # For image processing
import matplotlib.pyplot as plt  # (Not used here, but kept if needed later)

# -------------------------------------------------
# MediaPipe Hand landmark initialization
# -------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand detection model
# static_image_mode=True → process images, not video stream
# min_detection_confidence=0.3 → confidence threshold
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

# -------------------------------------------------
# Dataset directory
# -------------------------------------------------
DATA_DIR = './data'

# Lists to store features and class labels
data = []
labels = []

# -------------------------------------------------
# Loop through each class folder
# Each folder name is treated as a label
# -------------------------------------------------
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        data_aux = []   # Stores landmark features for one image

        x_ = []         # Stores all x-coordinates of landmarks
        y_ = []         # Stores all y-coordinates of landmarks

        # Read image from disk
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))

        # Convert BGR to RGB (required by MediaPipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(img_rgb)

        # If at least one hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # -----------------------------------------
                # Step 1: Collect raw x and y coordinates
                # -----------------------------------------
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # -----------------------------------------
                # Step 2: Normalize landmarks
                # (makes model position-invariant)
                # -----------------------------------------
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # Subtract minimum x and y values
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Store features and corresponding label
            data.append(data_aux)
            labels.append(dir_)

# -------------------------------------------------
# Save extracted features and labels using pickle
# -------------------------------------------------
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
