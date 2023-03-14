import cv2
import numpy as np


# Load the Aruco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters
aruco_params = cv2.aruco.DetectorParameters_create()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the frame
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)

    # If the marker with ID 23 is detected, estimate its pose
    if ids is not None:
        if 23 in ids:
            index = np.where(ids == 23)[0][0]
            marker_corners = corners[index][0]
            center = np.mean(marker_corners, axis=0).astype(int)

            # Print out the center position
            print("Marker center position:", center)

        # Draw the detected markers and IDs on the frame
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Print the IDs of the detected markers
        print(ids)

    # Display the output frame
    cv2.imshow('Output', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
