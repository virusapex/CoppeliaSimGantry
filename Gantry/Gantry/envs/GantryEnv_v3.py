import time
import numpy as np
import gymnasium as gym
import cv2
import psutil
import sys
import json
import os
from gymnasium import spaces, logger
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from Gantry.envs.GantrySimModel_v3 import GantrySimModel


class GantryEnv(gym.Env):
    """
    ## Description
    "Gantry" is a two-jointed robot system. The goal is to move the robot's end effector close to a
    target that is spawned at a random position.

    ## Observation Space
    Observations consist of

    - The coordinates of the ArUco marker
    - The coordinates of target position
    - The velocities of the moving platforms
    - The distance to the target position in both axes
    - The cosine between vectors of current movement and vector leading to wanted target (similarity)

    The observation is a `ndarray` with shape `(13,)` where the elements correspond to the following:

    |  Num | Observation                                    | Min  | Max | Unit                  |
    |  --- | ---------------------------------------------- | ---- | --- | --------------------- |
    |  0   | x-coordinate of the marker[42]                 |   0  | Inf | position (pixel)      |
    |  1   | y-coordinate of the marker[42]                 |   0  | Inf | position (pixel)      |
    |  2   | x-coordinate of the marker[43]                 |   0  | Inf | position (pixel)      |
    |  3   | y-coordinate of the marker[43]                 |   0  | Inf | position (pixel)      |
    |  4   | x-coordinate of the marker[44]                 |   0  | Inf | position (pixel)      |
    |  5   | y-coordinate of the marker[44]                 |   0  | Inf | position (pixel)      |
    |  6   | x-coordinate of the target                     |   0  | Inf | position (pixel)      |
    |  7   | y-coordinate of the target                     |   0  | Inf | position (pixel)      |
    |  8   | velocity of the x-platform                     | -Inf | Inf | velocity (pixel/step) |
    |  9   | velocity of the y-platform                     | -Inf | Inf | velocity (pixel/step) |
    |  10  | x-value of position_marker - position_target   | -Inf | Inf | position (pixel)      |
    |  11  | y-value of position_marker - position_target   | -Inf | Inf | position (pixel)      |
    |  12  | similarity value                               | -Inf | Inf | unitless              |
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None):
        """
        :param render_mode: Visualization type from Gymnasium
        :type render_mode: str
        """
        self.tags_last = {}
        self.avg_pos = np.zeros(2)
        self.avg_pos_last = np.zeros(2)
        self.dt = 0.0333  # time step in simulation seconds

        # Pixel position for wanted target
        self.target_position = np.zeros(2)

        # Don't forget to normalize when training
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(13,), dtype=np.float64)

        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(13,))
        self.state[:8] = self.np_random.integers(low=0, high=5, size=(8,))
        self.state[10:12] = self.np_random.integers(low=-5, high=5, size=(2,))
        self.counts = 0
        self.steps_beyond_done = None

        # Define the port range and status
        # TODO Add Established ports to the list for hyperparameter tuning
        if not os.path.exists("ports.json"):
            with open("ports.json", "w") as f:
                port_range = [port for port in range(23000, 23021, 2)]
                json.dump(port_range, f, indent=2)

        with open("ports.json", 'r+') as f:
            file = json.load(f)
            self.port = file.pop(0)
            f.seek(0)
            json.dump(file, f, indent=2)
            f.truncate()
        # status = 'ESTABLISHED'
        # ip_address = '127.0.0.1'
        # # Loop through each process and check if it matches the criteria
        # for conn in psutil.net_connections():
        #     try:
        #         if conn.status == status and conn.raddr.port in port_range and conn.raddr.ip == ip_address:
        #             print(f"A process is using port {conn.raddr.port} with status '{conn.status}'")
        #             port_range.remove(conn.raddr.port)

        #     except (psutil.AccessDenied, psutil.ZombieProcess):
        #         pass
        # try:
        #     port = port_range[0]
        # except IndexError:
        #     print("No available ports! Exiting...")
        #     sys.exit()

        # Connect to CoppeliaSim
        self.client = RemoteAPIClient(port=self.port)
        self.sim = self.client.getObject('sim')
        print(f'Connected to remote API server {self.port}.')
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
 
        self.position_history = []  # empty list to store previous positions
        self.v = [0.0, 0.0]
        self.min_distance = 760
        # For visualization purposes
        self.distance = 0
        self.reward = 0
        self.cosine_sim = 0

        self.gantry_sim_model = GantrySimModel()
        self.gantry_sim_model.initializeSimModel(self.sim)

    def step(self, action):
        """
        Single step of simulation in CoppeliaSim
        :param action: Value, generated by Gymnasium, theoretically the best action for current state
        :type action: nd.array
        """
        marker = 1  # Assume that markers are visible
        aruco = [42, 43, 44]  # IDs of ArUco

        # Position of ArUco tags
        tags = self.gantry_sim_model.getGantryPixelPosition(
                self.sim, self.visionSensorHandle, self.dist_coeffs)
        # Create dictionary with only 3 tags that represent suctionPad
        tags_history = {key:tags[key] for key in aruco}
        tags_target = tags[0]
        self.position_history.append(tags_history)  # add them to a delay buffer

        # TODO Change this into states of dictionary
        aruco_states = [False, False, False]

        # If history has more than 5 positions (165ms delay)
        if len(self.position_history) > 5:
            # Take the latest available set
            tags = self.position_history.pop(0)

            # Case where all markers are visible
            if all(all(tags[id]) for id in aruco):
                # Setting the target in case it isn't already
                if not all(self.target_position):
                    self.target_position = tags_target if all(tags_target) else self.target_position
        
                self.avg_pos      = (tags[aruco[0]] + tags[aruco[1]] + tags[aruco[2]]) / 3
                self.avg_pos_last = (self.tags_last[aruco[0]] + \
                                     self.tags_last[aruco[1]] + \
                                     self.tags_last[aruco[2]]) / 3
                self.v = [(self.avg_pos[0] - self.avg_pos_last[0])/(self.dt*1000),
                          (self.avg_pos[1] - self.avg_pos_last[1])/(self.dt*1000)]

            # Case where some markers are visible
            elif any(all(tags[id]) for id in aruco):
                # Setting the target in case it isn't already
                if not all(self.target_position):
                    self.target_position = tags_target if all(tags_target) else self.target_position

                # Find which markers are visible
                aruco_states = [all(tags[id]) for id in aruco]
                sum_pos = 0
                n = 0
                # This loop allows us to find average position on visible markers
                for id,state in enumerate(aruco_states):
                    if state:
                        sum_pos += tags[aruco[id]]
                        n += 1
                self.avg_pos = sum_pos / n

                # Find which markers were visible last time
                aruco_states = [all(self.tags_last[id]) for id in aruco]
                sum_pos = 0
                n = 0
                for id,state in enumerate(aruco_states):
                    if state:
                        sum_pos += self.tags_last[aruco[id]]
                        n += 1
                try:
                    self.avg_pos_last = sum_pos / n
                    self.v = [(self.avg_pos[0] - self.avg_pos_last[0])/(self.dt*1000),
                              (self.avg_pos[1] - self.avg_pos_last[1])/(self.dt*1000)]
                except ZeroDivisionError:
                    pass

            # Case where we can't see any of the markers
            else:
                # Setting the target in case it isn't already
                if not all(self.target_position):
                    self.target_position = tags_target if all(tags_target) else self.target_position

                marker = 0

            self.tags_last.update(tags)

        # Compute the distance between the current position and the target position
        distance = np.linalg.norm(np.array(self.avg_pos) - np.array(self.target_position))
        distance_last = np.linalg.norm(np.array(self.avg_pos_last) - np.array(self.avg_pos))
        vector_xy = np.array(self.target_position) - np.array(self.avg_pos)
        vector_last_xy = np.array(self.avg_pos) - np.array(self.avg_pos_last)
        
        self.cosine_sim = np.dot(vector_xy, vector_last_xy)/np.dot(distance, distance_last)
        if np.isnan(self.cosine_sim):
            self.cosine_sim = 0

        if distance < self.min_distance:
            self.min_distance = distance

        done = False

        # No matter the condition of markers, we grant the system the reward
        # for distance in the sim, so it should react to the situation
        # when final position is obstructed by the gantry
        distance_sim = self.gantry_sim_model.getDistanceToTarget(self.sim)
        reward = distance_sim and 1/distance_sim or 1000

        # Define the regularization parameter lambda
        lambda_ = np.mean(self.v) * 50

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

        # Set action
        action /= 2  # from [-1,1] to [-0.5,0.5]
        self.gantry_sim_model.setGantryVelocity(self.sim, action)

        self.state = np.array((tags[aruco[0]][0], tags[aruco[0]][1],
                               tags[aruco[1]][0], tags[aruco[1]][1],
                               tags[aruco[2]][0], tags[aruco[2]][1],
                               self.target_position[0], self.target_position[1],
                               self.v[0], self.v[1],
                               vector_xy[0], vector_xy[1], self.cosine_sim))

        self.counts += 1
        self.distance = distance
        self.reward = reward

        self.client.step()

        return self.state, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.counts = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(13,))
        self.state[:8] = self.np_random.integers(low=0, high=5, size=(8,))
        self.state[10:12] = self.np_random.integers(low=-5, high=5, size=(2,))
        self.steps_beyond_done = None

        # Create random distortion coefficients
        k1 = np.random.uniform(-0.2, 0.)
        k2 = np.random.uniform(0., 0.1)
        p1 = np.random.uniform(-0.01, 0.)
        p2 = np.random.uniform(-0.01, 0.)
        k3 = 0.0
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])

        self.target_position = np.zeros(2)

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)  # ensure the Coppeliasim is stopped

        self.client.setStepping(True)
        self.sim.startSimulation()
        self.gantry_sim_model.setGantryVelocity(self.sim, [0.0, 0.0])
        self.gantry_sim_model.resetGantryPosition(self.sim)
        self.gantry_sim_model.resetCameraOrientation(self.sim, self.visionSensorHandle)
        self.gantry_sim_model.resetTargetPosition(self.sim)

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

        tags = [42, 43, 44]  # 3 tags on end effector
        centers = {}
        # If the marker with any ID is detected, estimate its center
        if ids is not None:
            for tag in tags:
                if tag in ids:
                    index = np.where(ids == tag)[0][0]
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
        if ids is not None and all(self.avg_pos) and all(self.target_position):
            cv2.line(img_cropped, self.target_position, self.avg_pos.astype(int), (0, 255, 0), 2)
            cv2.circle(img_cropped, self.target_position, 15, (0, 0, 255), 1)

        if self.render_mode == "human":
            # Display the original and cropped images side by side
            cv2.imshow('Original', img)
            cv2.imshow('Cropped', img_cropped)
            cv2.waitKey(20)
        else:
            return img_cropped

    def close(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)  # ensure the Coppeliasim is stopped

        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        with open("ports.json", 'r+') as f:
            file = json.load(f)
            file.insert(0,self.port)
            f.seek(0)
            json.dump(file, f, indent=2)
            f.truncate()
        cv2.destroyAllWindows()
        return None


if __name__ == "__main__":
    env = GantryEnv(render_mode="human")
    env.reset()

    for _ in range(500):
        action = env.action_space.sample()  # random action
        env.step(action)
        env.render()

    env.close()
