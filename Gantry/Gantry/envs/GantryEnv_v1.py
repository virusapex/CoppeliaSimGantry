import time
import numpy as np
import gym
import cv2
import psutil
import sys
from gym.utils import seeding
from gym import spaces, logger
from zmqRemoteApi import RemoteAPIClient
from Gantry.envs.GantrySimModel import GantrySimModel


class GantryEnv(gym.Env):
    """
    ## Description
    "Gantry" is a two-jointed robot system. The goal is to move the robot's end effector close to a
    target that is spawned at a random position.

    ## Observation Space
    Observations consist of

    - The coordinates of the Aruco marker
    - The coordinates of target position
    - The velocities of the moving platforms
    - The distance to the target position in both axes
    - The cosine between vectors of current movement and vector leading to wanted target (similarity)

    The observation is a `ndarray` with shape `(7,)` where the elements correspond to the following:

    | Num | Observation                                    | Min  | Max | Unit                  |
    | --- | ---------------------------------------------- | ---- | --- | --------------------- |
    | 0   | x-coordinate of the marker                     |   0  | Inf | position (pixel)      |
    | 1   | y-coordinate of the marker                     |   0  | Inf | position (pixel)      |
    | 2   | x-coordinate of the target                     |   0  | Inf | position (pixel)      |
    | 3   | y-coordinate of the target                     |   0  | Inf | position (pixel)      |
    | 4   | velocity of the x-platform                     | -Inf | Inf | velocity (pixel/step) |
    | 5   | velocity of the y-platform                     | -Inf | Inf | velocity (pixel/step) |
    | 6   | x-value of position_marker - position_target   | -Inf | Inf | position (pixel)      |
    | 7   | y-value of position_marker - position_target   | -Inf | Inf | position (pixel)      |
    | 8   | similarity value                               | -Inf | Inf | unitless              |
    """

    metadata = {'render.modes': ['human', 'rgb_array'],
                'render_fps':   50}

    def __init__(self, render_mode=None):
        super(GantryEnv, self).__init__()
        self.q_last = [0, 470]

        # Pixel limits for wanted target
        self.x_max = 480
        self.x_min = 150
        self.y_max = 320
        self.y_min = 150

        # Don't forget to normalize when training
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(9,), dtype=np.float64)

        self.seed()

        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(9,))
        self.state[:4] = self.np_random.randint(low=0, high=5, size=(4,))
        self.state[6:8] = self.np_random.randint(low=-5, high=5, size=(2,))
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
        self.v = [0.0, 0.0]
        self.min_distance = 760
        # For visualization purposes
        self.distance = 0
        self.reward = 0
        self.cosine_sim = 0

        self.gantry_sim_model = GantrySimModel()
        self.gantry_sim_model.initializeSimModel(self.sim)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        dt = 0.0333  # time step in simulation seconds
        marker = 1

        # Position of Gantry robot (X- and Y-axis)
        q = self.gantry_sim_model.getGantryPixelPosition(
            self.sim, self.visionSensorHandle, self.dist_coeffs)
        # TODO Probably remove since gSDE is present
        # q += np.random.randint(-3,3,2)  # simulating camera noise

        if q[0] == 0.0:
            marker = 0
            q = self.q_last  # marker was not found
            if len(self.position_history) > 0:  # if history exists
                self.q_last = self.position_history[0]
            else:
                self.q_last = [0, 470]
        else:
            self.position_history.append(q)
            if len(self.position_history) > 5:  # if history has more than 5 positions (165ms delay)
                # Remove the oldest position and set it as current position
                q = self.position_history.pop(0)
                self.v = [(q[0] - self.q_last[0])/(dt*1000),   # velocity change for dt
                          (q[1] - self.q_last[1])/(dt*1000)]

        # Set action
        action /= 2  # from [-1,1] to [-0.5,0.5]
        self.gantry_sim_model.setGantryVelocity(self.sim, action)

        # Compute the distance between the current position and the target position
        distance = np.linalg.norm(np.array(q) - np.array(self.wanted_pixel))
        distance_last = np.linalg.norm(
            np.array(self.q_last) - np.array(q))
        vector_xy = np.array(self.wanted_pixel) - np.array(q)
        vector_last_xy = np.array(q) - np.array(self.q_last)
        cosine_sim = np.dot(vector_xy, vector_last_xy)/np.dot(distance, distance_last)
        if np.isnan(cosine_sim):
            cosine_sim = 0

        if distance < self.min_distance:
            self.min_distance = distance

        done = False

        # if distance_decreasing:
        if not marker:
            reward = -1
            cosine_sim = 0
            vector_xy = np.array([500, 500])
            self.v = np.array([0.0, 0.0])

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

        self.state = (q[0], q[1], self.wanted_pixel[0], self.wanted_pixel[1],
                      self.v[0], self.v[1], vector_xy[0], vector_xy[1], cosine_sim)
        self.counts += 1
        self.distance = distance
        self.reward = reward
        self.cosine_sim = cosine_sim

        self.client.step()

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.counts = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(9,))
        self.state[:4] = self.np_random.randint(low=0, high=5, size=(4,))
        self.state[6:8] = self.np_random.randint(low=-5, high=5, size=(2,))
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

        return np.array(self.state, dtype=np.float32)

    def render(self, mode):
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
