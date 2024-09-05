# Import the os module for interacting with the operating system (e.g., file paths).
import os

# Import the cv2 module from OpenCV for capturing video and handling image processing tasks.
import cv2

# Define the directory where the data will be stored.
DATA_DIR = './data'

# Check if the DATA_DIR directory exists. If it doesn't, create it.
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes/categories for which data will be collected.
number_of_classes = 8

# Define the number of images to capture for each class.
dataset_size = 100

# Initialize the video capture object to access the webcam (device 0).
cap = cv2.VideoCapture(0)

# Loop over each class to collect data for that class.
for j in range(number_of_classes):
    # Check if the directory for the current class exists. If it doesn't, create it.
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Print a message indicating which class is currently being collected.
    print('Collecting data for class {}'.format(j))

    done = False  # This variable is not used in the loop but could be for additional control.

    # Loop until the user is ready to start collecting data for the current class.
    while True:
        # Capture a frame from the webcam.
        ret, frame = cap.read()
        
        # Display a message on the frame prompting the user to press 'V' when ready.
        cv2.putText(frame, 'Ready? Press "V" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Show the frame with the prompt in a window titled 'frame'.
        cv2.imshow('frame', frame)
        
        # Check if the user presses the 'V' key. If so, break out of the loop and start collecting data.
        if cv2.waitKey(25) == ord('V'):
            break

    # Initialize a counter to keep track of the number of images collected for the current class.
    counter = 0
    
    # Continue capturing images until the desired dataset_size is reached.
    while counter < dataset_size:
        # Capture a frame from the webcam.
        ret, frame = cap.read()
        
        # Display the frame in the 'frame' window.
        cv2.imshow('frame', frame)
        
        # Wait for 25 milliseconds between frames.
        cv2.waitKey(25)
        
        # Save the captured frame as a .jpg file in the corresponding class directory.
        # The file is named with the counter value (e.g., '0.jpg', '1.jpg').
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        # Increment the counter to track how many images have been saved.
        counter += 1

# Release the video capture object when done to free the webcam for other applications.
cap.release()

# Close all OpenCV windows that were opened during the execution.
cv2.destroyAllWindows()
