import serial
import random
import numpy as np
import cv2
import pyrealsense2 as rs
import time


class GantryRealModel():

    def __init__(self):

        # Load the Aruco dictionary
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

        # Initialize the detector parameters
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Realsense set-up
        self.pipeline = rs.pipeline()
        config = rs.config()
        # Enable RGB camera
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # Initializing serial connection
        self.ser = serial.Serial('/dev/ttyACM0', baudrate=250000)
        time.sleep(3)

    def close(self):
        self.pipeline.stop()
        self.ser.close()

    def getGantryPixelPosition(self):

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert the color image to a numpy array
        img = np.asanyarray(color_frame.get_data())

        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the markers in the frame
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        # If the marker with ID 23 is detected, estimate its pose
        if ids is not None:
            if 284 in ids:
                index = np.where(ids == 284)[0][0]
                marker_corners = corners[index][0]
                center = np.mean(marker_corners, axis=0).astype(int)

                # Return the center position
                return center, corners, img

        return [0, 720], None, img

    def setGantryVelocity(self, speed):
        dir_y = 0 if speed[1] < 0. else 1
        dir_x = 0 if speed[0] < 0. else 1
        speed_y = int(abs(speed[1])*1000)
        speed_x = int(abs(speed[0])*1000)
        # Message syntax: #DIR_Y;SPEED_Y;DIR_X;SPEED_X;
        # speed is in mm/sec, dir 0 is negative, 1 is positive
        # e.g. #1;10;0;15;
        self.ser.write(f"#{dir_x};{speed_x};{dir_y};{speed_y};".encode())
        #self.ser.read(100)
