import pickle                   # For loading trained model
import cv2                      # OpenCV for webcam & drawing
import mediapipe as mp          # MediaPipe for hand landmarks
import numpy as np              # For numerical arrays

# -------------------------------------------------
# Load trained Random Forest model
# -------------------------------------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# -------------------------------------------------
# Open webcam
# -------------------------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------------------------
# MediaPipe Hand setup
# -------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# -------------------------------------------------
# Label mapping
# -------------------------------------------------
labels_dict = {
    0: 'A',
    1: 'B',
    2: 'L'
}

THRESHOLD = 0.7   # ⭐ confidence threshold

# -------------------------------------------------
# Real-time prediction loop
# -------------------------------------------------
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        # Bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # -------------------------------------------------
        # Prediction with confidence check
        # -------------------------------------------------
        probs = model.predict_proba([np.asarray(data_aux)])
        confidence = np.max(probs)
        predicted_class = np.argmax(probs)

        if confidence < THRESHOLD:
            predicted_character = "None"
        else:
            predicted_character = labels_dict[predicted_class]

        # -------------------------------------------------
        # Draw result
        # -------------------------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(
            frame,
            f"{predicted_character} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------------------------
# Release resources
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()
