# Import the pickle module for loading the saved model.
import pickle

# Import OpenCV for capturing video and handling image processing tasks.
import cv2

# Import MediaPipe for hand tracking and image processing.
import mediapipe as mp

# Import NumPy for numerical operations and array handling.
import numpy as np

# Load the trained model from the 'model.p' file.
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe components for hand tracking.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure the MediaPipe Hands solution for static image mode.
# - static_image_mode=True: Used for processing static images rather than continuous video.
# - min_detection_confidence=0.3: The minimum confidence level required for detecting hands.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create a dictionary mapping model output labels to corresponding hand signs.
labels_dict = pairs = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'I LOVE YOU', 5: 'GOOD JOB', 6: 'HELLO', 7: 'EAT'}

# Define a generator function that captures video frames from the webcam and processes them.
def generate_frames():
    # Open the webcam for capturing video.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Continuously capture and process video frames.
    while True:
        data_aux = []  # List to store normalized hand landmark coordinates for prediction.
        x_ = []  # List to store x-coordinates of hand landmarks.
        y_ = []  # List to store y-coordinates of hand landmarks.

        # Capture a frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            continue

        # Get the height and width of the frame.
        H, W, _ = frame.shape

        # Convert the frame from BGR (OpenCV format) to RGB (MediaPipe format).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands to detect hand landmarks.
        results = hands.process(frame_rgb)
        
        # If hand landmarks are detected:
        if results.multi_hand_landmarks:
            # Loop through each detected hand and draw landmarks on the frame.
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # Image to draw landmarks on.
                    hand_landmarks,  # Hand landmarks detected by MediaPipe.
                    mp_hands.HAND_CONNECTIONS,  # Hand connections to draw.
                    mp_drawing_styles.get_default_hand_landmarks_style(),  # Landmark style.
                    mp_drawing_styles.get_default_hand_connections_style())  # Connection style.

            # Loop again to extract and normalize the hand landmark coordinates.
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Get the x-coordinate of the landmark.
                    y = hand_landmarks.landmark[i].y  # Get the y-coordinate of the landmark.

                    x_.append(x)  # Append the x-coordinate to the list.
                    y_.append(y)  # Append the y-coordinate to the list.

                # Normalize the coordinates by subtracting the minimum x and y values.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Normalize and append x-coordinate.
                    data_aux.append(y - min(y_))  # Normalize and append y-coordinate.

            # Calculate the bounding box for the detected hand.
            x1 = int(min(x_) * W) - 10  # Left x-coordinate of the bounding box.
            y1 = int(min(y_) * H) - 10  # Top y-coordinate of the bounding box.
            x2 = int(max(x_) * W) - 10  # Right x-coordinate of the bounding box.
            y2 = int(max(y_) * H) - 10  # Bottom y-coordinate of the bounding box.

            try:
                # Use the trained model to predict the hand sign based on the landmarks.
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]  # Map prediction to the corresponding label.
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_character = "?"  # If prediction fails, use "?" as the label.

            # Draw the bounding box around the detected hand.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            # Display the predicted hand sign above the bounding box.
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Encode the frame as a JPEG image.
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()  # Convert the encoded image to bytes.

        # Yield the frame in a format suitable for streaming in a web application.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the webcam when done.
    cap.release()
    # Close any OpenCV windows that were opened during execution.
    cv2.destroyAllWindows()
