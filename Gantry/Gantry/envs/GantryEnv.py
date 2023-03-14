import numpy as np
import gym
from gym.utils import seeding
from gym import spaces, logger
import time
from zmqRemoteApi import RemoteAPIClient
from Gantry.envs.GantrySimModel import GantrySimModel


class GantryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, port):
        super(GantryEnv, self).__init__()
        self.q_last = [0.0, 0.0]

        self.x_max = 580
        self.x_min = 40
        self.y_max = 420
        self.y_min = 40

        high = np.array(
            [
                self.x_max,
                np.finfo(np.float32).max,
                self.y_max,
                np.finfo(np.float32).max
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                self.x_min,
                np.finfo(np.float32).min,
                self.y_min,
                np.finfo(np.float32).min
            ],
            dtype=np.float32,
        )

        # Don't forget to normalize when training
        self.action_space = spaces.Box(low=0.0, high=0.6, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.seed()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.counts = 0
        self.steps_beyond_done = None

        # Connect to CoppeliaSim
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.getObject('sim')
        print('Connected to remote API server.')
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.client.setStepping(True)
        self.sim.startSimulation()

        self.visionSensorHandle = self.sim.getObject('/kinect/rgb')
        print('Connected to vision sensor.')
        # Initialize distortion coefficients
        self.dist_coeffs = np.zeros((5,))
        # Wanted pixel position
        self.wanted_pixel = [350, 250]
        self.position_history = []  # empty list to store previous positions
        self.v = [0.0, 0.0]

        self.gantry_sim_model = GantrySimModel()
        self.gantry_sim_model.initializeSimModel(self.sim)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        q = [0.0, 0.0]
        dt = 0.005  # time step in simulation seconds

        # Position of Gantry robot (X- and Y-axis)
        q[0], q[1] = self.gantry_sim_model.getGantryPixelPosition(
            self.sim, self.visionSensorHandle, self.dist_coeffs)

        self.position_history.append(q)
        if len(self.position_history) > 5:  # if history has more than 5 positions (25ms delay)
            q = self.position_history.pop(0)  # remove oldest position and set it as current position
            self.v = [(q[0] - self.q_last[0])/dt,   # velocity change for dt
                      (q[1] - self.q_last[1])/dt]
            self.q_last = q

        # Set action
        self.gantry_sim_model.setGantryPosition(self.sim, action)

        # Compute the distance between the current position and the target position
        distance = np.linalg.norm(np.array([q]) - np.array(self.wanted_pixel))

        # Conditions for stopping the episode
        done = (q[0] <= self.x_min) or (q[0] >= self.x_max) \
            or (q[1] <= self.y_min) or (q[1] >= self.y_max)
        done = bool(done)

        if distance < 1.0:
            # Maximum reward if the robot is within 1.0 units of the target position
            reward = 100.0
        else:
            # Reward is inversely proportional to the distance from the target position
            reward = 1.0 / distance

        # Define the regularization parameter lambda
        lambda_ = 0.00001

        # Compute the L2 norm of the parameter vector theta
        reg_term = lambda_ * np.linalg.norm(self.v) ** 2

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

        self.state = (q[0], self.v[0], q[1], self.v[1])
        self.counts += 1

        self.client.step()

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.counts = 0
        self.push_force = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
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
        self.gantry_sim_model.setGantryPosition(self.sim, [0, 0])

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

    for _ in range(1000):
        action = env.action_space.sample()  # random action
        env.step(action)
        print(env.state)

    env.close()
