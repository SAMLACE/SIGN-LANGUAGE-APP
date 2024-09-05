# Import the os module for interacting with the operating system (e.g., file paths).
import os

# Import the pickle module for serializing and saving data objects.
import pickle

# Import the MediaPipe library for hand tracking and image processing.
import mediapipe as mp

# Import OpenCV for image processing and computer vision tasks.
import cv2

# Import Matplotlib for plotting (although it's not used in this script).
import matplotlib.pyplot as plt

# Initialize MediaPipe's hand tracking model.
mp_hands = mp.solutions.hands

# Initialize MediaPipe's drawing utilities to visualize hand landmarks.
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe's drawing styles for styling hand landmarks and connections.
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the MediaPipe Hands solution in static image mode.
# - static_image_mode=True: Used for processing static images rather than video.
# - min_detection_confidence=0.3: The minimum confidence level required for detecting hands.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory where the image data is stored.
DATA_DIR = './data'

# Initialize empty lists to store the processed hand landmark data and corresponding labels.
data = []
labels = []

# Loop through each directory in the DATA_DIR (each representing a different class).
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image file in the current class directory.
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Auxiliary list to store the landmarks of a single image.

        x_ = []  # List to store x-coordinates of all hand landmarks in the image.
        y_ = []  # List to store y-coordinates of all hand landmarks in the image.

        # Read the image from the file.
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        
        # Convert the image from BGR (OpenCV format) to RGB (MediaPipe format).
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands to detect hand landmarks.
        results = hands.process(img_rgb)
        
        # If hand landmarks are detected in the image:
        if results.multi_hand_landmarks:
            # Loop through each detected hand (though in this context, it's likely one hand per image).
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through each landmark point in the hand.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Get the x-coordinate of the landmark.
                    y = hand_landmarks.landmark[i].y  # Get the y-coordinate of the landmark.

                    x_.append(x)  # Add the x-coordinate to the list.
                    y_.append(y)  # Add the y-coordinate to the list.

                # Normalize the landmarks by subtracting the minimum x and y values from each landmark.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize and add the x-coordinate to the auxiliary list.
                    data_aux.append(y - min(y_))  # Normalize and add the y-coordinate to the auxiliary list.

            # Add the processed landmarks for this image to the data list.
            data.append(data_aux)
            
            # Add the class label (directory name) for this image to the labels list.
            labels.append(dir_)

# Open a file named 'data.pickle' in binary write mode to store the serialized data.
f = open('data.pickle', 'wb')

# Use pickle to serialize the data and labels, and save them to the file.
pickle.dump({'data': data, 'labels': labels}, f)

# Close the file after writing to it.
f.close()
