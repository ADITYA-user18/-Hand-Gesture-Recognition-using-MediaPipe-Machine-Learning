import os              # For directory and file handling
import cv2             # OpenCV library for camera and image processing

# -------------------------------
# Base directory where data will be stored
# -------------------------------
DATA_DIR = './data'

# Create the data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# -------------------------------
# Configuration
# -------------------------------
number_of_classes = 3     # Number of gesture/classes to collect
dataset_size = 100        # Number of images per class

# -------------------------------
# Initialize webcam
# (Change index if camera doesn't open: 0, 1, 2...)
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# Loop through each class
# -------------------------------
for j in range(number_of_classes):

    # Create a folder for each class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # -------------------------------
    # Wait for user to press 'Q' to start capturing
    # -------------------------------
    while True:
        ret, frame = cap.read()   # Read a frame from webcam

        # Display instructions on screen
        cv2.putText(
            frame,
            'Ready? Press "Q" ! :)',
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        cv2.imshow('frame', frame)

        # Break when user presses 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # -------------------------------
    # Capture and save images
    # -------------------------------
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()   # Capture frame
        cv2.imshow('frame', frame)

        # Save image in class folder
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)

        cv2.waitKey(25)
        counter += 1

# -------------------------------
# Release resources
# -------------------------------
cap.release()             # Release webcam
cv2.destroyAllWindows()   # Close all OpenCV windows
