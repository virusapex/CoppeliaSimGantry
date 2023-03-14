from zmqRemoteApi import RemoteAPIClient
import numpy as np
import cv2


# Connect to CoppeliaSim
client = RemoteAPIClient(port=23000)
sim = client.getObject('sim')
print('Connected to remote API server.')

visionSensorHandle = sim.getObject('/kinect/rgb')
print('Connected to vision sensor.')

# Load the Aruco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters
aruco_params = cv2.aruco.DetectorParameters_create()

# Set camera parameters
width = 640
height = 480
fov = 90
focal_length = 2.0

# Create camera matrix
fx = fy = (width / 2) / np.tan(np.deg2rad(fov / 2))
cx = width / 2
cy = height / 2
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Create distortion coefficients
k1 = np.random.uniform(-0.3, -0.2)
k2 = np.random.uniform(0.1, 0.2)
p1 = np.random.uniform(-0.01, -0.005)
p2 = np.random.uniform(-0.01, -0.005)
k3 = 0.0
dist_coeffs = np.array([k1, k2, p1, p2, k3])

# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)
client.setStepping(True)
sim.startSimulation()

while (t := sim.getSimulationTime()) < 3:
    img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)

    # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
    # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
    # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

    # Apply lens distortion
    img_distorted = cv2.undistort(img, camera_matrix, dist_coeffs)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img_distorted, cv2.COLOR_BGR2GRAY)

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
        img_distorted = cv2.aruco.drawDetectedMarkers(img_distorted, corners, ids)

        # Print the IDs of the detected markers
        print(ids)

    # Cropping
    img_cropped = img_distorted[10:470, 10:630]

    # Display the original and distorted images side by side
    cv2.imshow('Original', img)
    cv2.imshow('Distorted', img_distorted)
    cv2.imshow('Cropped', img_cropped)
    cv2.waitKey(1)
    client.step()  # triggers next simulation step

sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

cv2.destroyAllWindows()

print('Program ended')
