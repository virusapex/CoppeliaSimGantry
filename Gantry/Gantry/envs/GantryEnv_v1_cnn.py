import time
import numpy as np
import gymnasium as gym
import cv2
import psutil
import sys
from gymnasium import spaces, logger
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from Gantry.envs.GantrySimModel import GantrySimModel


class GantryEnv(gym.Env):
    """
    ## Description
    "Gantry" is a two-jointed robot system. The goal is to move the robot's end effector close to a
    target that is spawned at a random position.

    ## Observation Space
    Observation in this case is a grayscale array with following shape:
    Box(0, 255, (460, 620), uint8)
    """

    def __init__(self, render_mode=None):
        super(GantryEnv, self).__init__()
        self.q_last = [0, 470]
        self.dt = 0.0333  # time step in simulation seconds
        self.metadata["render_modes"] = ['human', 'rgb_array'],
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # Pixel limits for wanted target
        self.x_max = 480
        self.x_min = 150
        self.y_max = 320
        self.y_min = 150

        # Don't forget to normalize when training
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(460, 620, 1), dtype=np.uint8)

        self.state = np.expand_dims(np.zeros((460, 620), np.uint8), axis=-1)
        self.counts = 0
        self.steps_beyond_done = None

        # Define the port range and status
        port_range = [port for port in range(23000, 23021, 2)]
        status = 'ESTABLISHED'
        ip_address = '127.0.0.1'

        # Loop through each process and check if it matches the criteria
        for conn in psutil.net_connections():
            try:
                if conn.status == status and conn.raddr.port in port_range and conn.raddr.ip == ip_address:
                    print(f"A process is using port {conn.raddr.port} with status '{conn.status}'")
                    port_range.remove(conn.raddr.port)

            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass
        try:
            port = port_range[0]
        except IndexError:
            print("No available ports! Exiting...")
            sys.exit()

        # Connect to CoppeliaSim
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.getObject('sim')
        print(f'Connected to remote API server {port}.')
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(
            self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.client.setStepping(True)
        self.sim.startSimulation()

        self.visionSensorHandle = self.sim.getObject('/rgb')
        print('Connected to vision sensor.')

        self.render_mode = render_mode
        # Initialize distortion coefficients
        self.dist_coeffs = np.zeros((5,))
        # Wanted pixel position
        self.wanted_pixel = [350, 250]
        self.position_history = []  # empty list to store previous positions
        self.min_distance = 760
        # For visualization purposes
        self.distance = 0
        self.reward = 0
        self.cosine_sim = 0

        self.gantry_sim_model = GantrySimModel()
        self.gantry_sim_model.initializeSimModel(self.sim)

    def step(self, action):
        marker = 1

        # Position of Gantry robot (X- and Y-axis + image)
        q, img = self.gantry_sim_model.getGantryPixelPosition(
            self.sim, self.visionSensorHandle, self.dist_coeffs, mode="raw")

        if q[0] == 0.0:
            marker = 0
            q = self.q_last  # marker was not found
            if len(self.position_history) > 0:  # if history exists
                self.q_last = self.position_history[0][0]
            else:
                self.q_last = [0, 470]
        else:
            self.position_history.append((q, img))
            if len(self.position_history) > 5:  # if history has more than 5 (image,position) tuples (165ms delay)
                # Remove the oldest tuple and set it as current state
                q, img = self.position_history.pop(0)

        # Set action
        action /= 2  # from [-1,1] to [-0.5,0.5]
        self.gantry_sim_model.setGantryVelocity(self.sim, action)

        # Compute the distance between the current position and the target position
        distance = np.linalg.norm(np.array(q) - np.array(self.wanted_pixel))

        if distance < self.min_distance:
            self.min_distance = distance

        done = False

        if not marker:
            reward = -1
        else:
            reward = distance and 760/distance or 1000
            self.q_last = q

        # Define the regularization parameter lambda
        lambda_ = 500

        # Compute the L2 norm of the parameter vector theta
        reg_term = lambda_ * (np.linalg.norm(action) ** 2)

        if not done:
            # Normalizing distance values
            reward -= reg_term
        elif self.steps_beyond_done is None:
            # Out of bounds
            self.steps_beyond_done = 0
            reward -= reg_term
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        self.state = np.expand_dims(img, axis=-1)
        self.counts += 1
        self.reward = reward

        self.client.step()

        return self.state, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.counts = 0
        self.state = np.expand_dims(np.zeros((460, 620), np.uint8), axis=-1)
        self.steps_beyond_done = None

        # Create random distortion coefficients
        k1 = np.random.uniform(-0.3, -0.2)
        k2 = np.random.uniform(0.1, 0.2)
        p1 = np.random.uniform(-0.01, -0.005)
        p2 = np.random.uniform(-0.01, -0.005)
        k3 = 0.0
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])

        self.wanted_pixel = [np.random.randint(self.x_min, self.x_max),
                             np.random.randint(self.y_min, self.y_max)]

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)  # ensure the Coppeliasim is stopped

        # Allows to turn off visualization
        # vrep_sim.simxSetBoolParam(
        #     self.cart_pole_sim_model.client_ID,
        #     vrep_sim.sim_boolparam_display_enabled,
        #     False,
        #     vrep_sim.simx_opmode_oneshot)

        self.client.setStepping(True)
        self.sim.startSimulation()
        self.gantry_sim_model.setGantryVelocity(self.sim, [0.0, 0.0])
        self.gantry_sim_model.resetGantryPosition(self.sim)
        self.gantry_sim_model.resetCameraOrientation(self.sim, self.visionSensorHandle)

        return self.state, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        img, resX, resY = self.sim.getVisionSensorCharImage(
                self.visionSensorHandle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)

        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

        # Apply lens distortion
        img_distorted = cv2.undistort(img, self.gantry_sim_model.camera_matrix,
                                      self.dist_coeffs)
        img_cropped = img_distorted[10:470, 10:630]

        # Convert the frame to grayscale
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

        # Detect the markers in the frame
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray, self.gantry_sim_model.aruco_dict,
            parameters=self.gantry_sim_model.aruco_params)

        # If the marker with ID 23 is detected, estimate its pose
        if ids is not None:
            if 23 in ids:
                index = np.where(ids == 23)[0][0]
                marker_corners = corners[index][0]
                center = np.mean(marker_corners, axis=0).astype(int)

            # Draw the detected markers and IDs on the frame
            img_cropped = cv2.aruco.drawDetectedMarkers(img_cropped, corners, ids)

        cv2.putText(img_cropped, f"Reward: {self.reward:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img_cropped, f"Distance (px): {self.distance:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img_cropped, f"Cosine similarity: {self.cosine_sim:.3f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if ids is not None:
            cv2.line(img_cropped, self.wanted_pixel, center, (0, 255, 0), 2)
        cv2.circle(img_cropped, self.wanted_pixel, 15, (0, 0, 255), 1)

        if self.render_mode == "human":
            # Display the original and cropped images side by side
            cv2.imshow('Original', img)
            cv2.imshow('Cropped', img_cropped)
            cv2.waitKey(20)
        else:
            return img_cropped

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        cv2.destroyAllWindows()
        return None


if __name__ == "__main__":
    env = GantryEnv(render_mode="human")
    env.reset()

    for _ in range(500):
        action = env.action_space.sample()  # random action
        env.step(action)
        env.render("human")

    env.close()
