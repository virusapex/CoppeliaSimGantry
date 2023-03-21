import random
import numpy as np
import cv2


class GantrySimModel():

    def __init__(self, name='Gantry'):
        self.name = name
        self.client_ID = None

        self.block_handle = None
        self.revolute_joint_handle = None
        # X and Y axes
        self.gantryJoints = [-1, -1]
        self.floor_handle = None

        # Load the Aruco dictionary
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

        # Initialize the detector parameters
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Set camera parameters
        width = 640
        height = 480
        fov = 90
        # focal_length = 2.0

        # Create camera matrix
        fx = fy = (width / 2) / np.tan(np.deg2rad(fov / 2))
        cx = width / 2
        cy = height / 2
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def initializeSimModel(self, sim):
        self.block_handle = sim.getObject('/End_effector')
        if (self.block_handle != -1):
            print('Got the block handle.')

        self.revolute_joint_handle = sim.getObject('/Z_axis')
        if (self.revolute_joint_handle != -1):
            print('Got the Z axis joint handle.')

        self.gantryJoints[0] = sim.getObject('./X_axis')
        self.gantryJoints[1] = sim.getObject('./Y_axis')
        if (self.gantryJoints[1] != -1):
            print('Got the X and Y joint handles.')

        q = sim.getObjectPosition(self.block_handle, -1)
        q = sim.getJointPosition(self.revolute_joint_handle)

        q = sim.getJointPosition(self.gantryJoints[0])
        q = sim.getJointPosition(self.gantryJoints[1])

        self.floor_handle = sim.getObject('/Floor/box')
        if (self.floor_handle != -1):
            print('Got the floor handle.')
        # Set the initialized position for each joint
        self.setGantryPosition(sim, [0, 0])

    def getGantryPosition(self, sim, joint_name):
        q = 0
        if joint_name == 'x':
            q = sim.getJointPosition(self.gantryJoints[0])
        elif joint_name == 'y':
            q = sim.getJointPosition(self.gantryJoints[1])
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return q

    def getGantryPixelPosition(self, sim, visionSensorHandle, dist_coeffs):
        img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)

        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

        # Apply lens distortion
        img_distorted = cv2.undistort(img, self.camera_matrix, dist_coeffs)

        # Cropping
        img_cropped = img_distorted[10:470, 10:630]

        # Convert the frame to grayscale
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

        # Detect the markers in the frame
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        # If the marker with ID 23 is detected, estimate its pose
        if ids is not None:
            if 23 in ids:
                index = np.where(ids == 23)[0][0]
                marker_corners = corners[index][0]
                center = np.mean(marker_corners, axis=0).astype(int)

                # Return the center position
                return center

        return [0.0, 470.0]

    def getJointPosition(self, sim, joint_name):
        q = 0
        if joint_name == 'block':
            q = sim.getObjectPosition(self.block_handle, -1)
        elif joint_name == 'revolute_joint':
            q = sim.getJointPosition(self.revolute_joint_handle)
        else:
            print('Error: joint name: \' ' + joint_name + '\' can not be recognized.')

        return q

    def setGantryPosition(self, sim, action):
        sim.setJointTargetPosition(self.gantryJoints[0], float(action[0]))
        sim.setJointTargetPosition(self.gantryJoints[1], float(action[1]))

    # TODO Add floor texture randomization
    def addRandomTexture(self, sim, texture_list):
        sim.setShapeTexture(self.floor_handle, random.choice(texture_list))
