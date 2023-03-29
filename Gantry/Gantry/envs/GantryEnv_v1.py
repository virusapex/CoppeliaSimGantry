import time
import numpy as np
import gym
from gym.utils import seeding
from gym import spaces, logger
from zmqRemoteApi import RemoteAPIClient
from Gantry.envs.GantrySimModel import GantrySimModel


class GantryEnv(gym.Env):
    """
        Observations consist of

        - The coordinates of the Aruco marker
        - The velocities of the moving platforms
        - The cosine between vectors of current movement and vector leading to wanted target (similarity)

        The observation is a `ndarray` with shape `(7,)` where the elements correspond to the following:
        
        | Num | Observation                                    | Min  | Max | Unit                  |
        | --- | ---------------------------------------------- | ---- | --- | --------------------- |
        | 0   | x-coordinate of the marker                     | -Inf | Inf | position (pixel)      |
        | 1   | y-coordinate of the marker                     | -Inf | Inf | position (pixel)      |
        | 2   | velocity of the x-platform                     | -Inf | Inf | velocity (pixel/step) |
        | 3   | velocity of the y-platform                     | -Inf | Inf | velocity (pixel/step) |
        | 4   | x-value of position_marker - position_target   | -Inf | Inf | position (m)          |
        | 5   | y-value of position_marker - position_target   | -Inf | Inf | position (m)          |
        | 6   | similarity value                               | -Inf | Inf | unitless              |
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, port):
        super(GantryEnv, self).__init__()
        self.q_last = [0, 470]

        self.x_max = 630
        self.x_min = 0
        self.y_max = 470
        self.y_min = 0

        # Don't forget to normalize when training
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(7,), dtype=np.float64)

        self.seed()

        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(7,))
        self.counts = 0
        self.steps_beyond_done = None

        # Connect to CoppeliaSim
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.getObject('sim')
        print('Connected to remote API server.')
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
        # Initialize distortion coefficients
        self.dist_coeffs = np.zeros((5,))
        # Wanted pixel position
        self.wanted_pixel = [350, 250]
        self.position_history = []  # empty list to store previous positions
        self.v = [0.0, 0.0]
        self.min_distance = 760

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

        if q[0] == 0.0:
            marker = 0
            q = self.q_last  # marker was not found
            if len(self.position_history) > 1:  # if history exists
                self.q_last = self.position_history[0]
            else:
                self.q_last = [0, 470]
        else:
            self.position_history.append(q)
            # TODO change delay buffer size
            if len(self.position_history) > 1:  # if history has more than 1 position (33ms delay)
                # remove oldest position and set it as current position
                q = self.position_history.pop(0)
                self.v = [(q[0] - self.q_last[0])/(dt*1000),   # velocity change for dt
                          (q[1] - self.q_last[1])/(dt*1000)]

        # Set action
        # action = (action + 1)/3.34  # from [-1,1] to [0,0.6]
        # self.gantry_sim_model.setGantryPosition(self.sim, action)
        action /= 2  # from [-1,1] to [-0.5,0.5]
        self.gantry_sim_model.setGantryVelocity(self.sim, action)

        # Compute the distance between the current position and the target position
        distance = np.linalg.norm(np.array(q) - np.array(self.wanted_pixel))
        distance_last = np.linalg.norm(
            np.array(self.q_last) - np.array(q))
        vector_xy = np.array(self.wanted_pixel) - np.array(q)
        vector_last_xy = np.array(q) - np.array(self.q_last)
        cosine_sim = np.dot(vector_xy,vector_last_xy)/np.dot(distance,distance_last)
        if np.isnan(cosine_sim):
            cosine_sim = 0

        if distance < self.min_distance:
            self.min_distance = distance

        # Conditions for stopping the episode
        # done = (q[0] < self.x_min) or (q[0] > self.x_max) \
        #     or (q[1] < self.y_min) or (q[1] > self.y_max)
        # done = bool(done)
        done = False

        # if distance_decreasing:
        if not marker:
            reward = 0
            cosine_sim = 0
            vector_xy = np.array([500,500])
            self.v = np.array([0.0,0.0])

        else:
            reward = 1 - (distance**2/617796)
            self.q_last = q
            if distance < 20.0:
                # Maximum reward if the robot is within 20 pixels of the target position
                reward = 10.0
            '''
            if distance < 20.0:
                # Maximum reward if the robot is within 20 pixels of the target position
                reward = 10.0
                self.wanted_pixel = [np.random.randint(self.x_min, self.x_max),
                                    np.random.randint(self.y_min, self.y_max)]
                self.counts = 0
            elif self.counts <= 200:
                # reward = -distance
                reward = 1 - (distance**2/617796)
            else:
                reward = 1 - (distance**2/617796)
                done = True
                self.reset()
            '''
        # Define the regularization parameter lambda
        lambda_ = 0.1

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

        self.state = (q[0], self.v[0], q[1], self.v[1],
                      vector_xy[0], vector_xy[1], cosine_sim)
        self.counts += 1

        self.client.step()

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.counts = 0
        self.push_force = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(7,))
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
        # self.gantry_sim_model.setGantryPosition(self.sim, [0.0, 0.0])
        self.gantry_sim_model.setGantryVelocity(self.sim, [0.0, 0.0])
        self.gantry_sim_model.resetGantryPosition(self.sim)
        self.gantry_sim_model.resetCameraOrientation(self.sim, self.visionSensorHandle)

        return np.array(self.state, dtype=np.float32)

    def render(self):
        return None

    def close(self):
        self.sim.stopSimulation()
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.defaultIdleFps)
        return None


if __name__ == "__main__":
    env = GantryEnv(23000)
    env.reset()

    for _ in range(500):
        action = env.action_space.sample()  # random action
        env.step(action)
        # print(env.state)

    env.close()