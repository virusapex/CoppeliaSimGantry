import time
import numpy as np
import gym
import cv2
from gym.utils import seeding
from gym import spaces, logger
from GantryRealModel import GantryRealModel


class GantryRealEnv(gym.Env):
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
        self.q_last = [0, 470]

        # Pixel limits for wanted target
        self.x_max = 640
        self.x_min = 500
        self.y_max = 500
        self.y_min = 200

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

        self.render_mode = render_mode

        # Wanted pixel position
        self.wanted_pixel = [350, 250]
        self.v = [0.0, 0.0]
        self.min_distance = 760
        # For visualization purposes
        self.distance = 0
        self.reward = 0
        self.cosine_sim = 0

        self.gantry_sim_model = GantryRealModel()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        dt = 0.0333  # time step in simulation seconds
        marker = 1

        # Position of Gantry robot (X- and Y-axis)
        q, self.corners, self.img = self.gantry_sim_model.getGantryPixelPosition()

        if q[0] == 0.0:
            marker = 0
            q = self.q_last  # marker was not found
        else:
            self.v = [(q[0] - self.q_last[0])/(dt*1000),   # velocity change for dt
                      (q[1] - self.q_last[1])/(dt*1000)]

        self.center = q

        # Set action
        action /= 2  # from [-1,1] to [-0.5,0.5]
        self.gantry_sim_model.setGantryVelocity(action)

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

        done = False

        # if distance_decreasing:
        if not marker:
            reward = -1
            cosine_sim = 0
            vector_xy = np.array([500,500])
            self.v = np.array([0.0,0.0])

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

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.counts = 0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(9,))
        self.state[:4] = self.np_random.randint(low=0, high=5, size=(4,))
        self.state[6:8] = self.np_random.randint(low=-5, high=5, size=(2,))
        self.steps_beyond_done = None

        self.wanted_pixel = [np.random.randint(self.x_min, self.x_max),
                             np.random.randint(self.y_min, self.y_max)]

        self.gantry_sim_model.setGantryVelocity([0.0, 0.0])

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
        if self.corners is not None:
            # Draw the detected markers and IDs on the frame
            img = cv2.aruco.drawDetectedMarkers(self.img, self.corners, 284)

        cv2.putText(img, f"Reward: {self.reward:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Distance (px): {self.distance:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Cosine similarity: {self.cosine_sim:.3f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.line(img, self.wanted_pixel, self.center, (0,255,0),2)
        cv2.circle(img, self.wanted_pixel, 15, (0,0,255),1)

        if self.render_mode == "human":
            # Display the original and cropped images side by side
            cv2.imshow('Original', img)
            cv2.waitKey(1)
        else:
            return img

    def close(self):
        cv2.destroyAllWindows()
        return None


if __name__ == "__main__":
    env = GantryRealEnv(render_mode="human")
    env.reset()

    for _ in range(500):
        start = time.time()
        action = env.action_space.sample()/10  # random action
        env.step(action)
        # env.render("human")
        end = time.time() - start
        print(end)

    env.close()
