import argparse
import numpy as np
import time
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO, TQC


parser = argparse.ArgumentParser()
parser.add_argument("--algo", help="RL Algorithm",
                    default="ppo", type=str, required=False)
parser.add_argument("-i", "--trained-agent", help="Path to a trained agent",
                    default="best_model", type=str)
parser.add_argument("--norm", type=str,
                    default="", help="Path to a VecNormalize statistics")
parser.add_argument("--env", type=str,
                    default="Gantry-v0", help="Environment ID")

args = parser.parse_args()


if args.env == "Gantry-v0":
    from Gantry.envs.GantryEnv import GantryEnv
    env = DummyVecEnv([lambda: GantryEnv(23006, render_mode="human")])
if args.env == "Gantry-v1":
    from GantryRealEnv_v1 import GantryRealEnv
    env = DummyVecEnv([lambda: GantryRealEnv()])
if args.norm:
    env = VecNormalize.load(args.norm, env)
else:
    env.norm_reward = False
env.training = False

if args.algo == "ppo":
    model = PPO.load(args.trained_agent, env=env)
elif args.algo == "ppo_lstm":
    model = RecurrentPPO.load(args.trained_agent, env=env)
elif args.algo == "sac":
    model = SAC.load(args.trained_agent, env=env)
elif args.algo == "tqc":
    model = TQC.load(args.trained_agent, env=env)

# Define the sigmoid function
def sigmoid(x, alpha=1):
    return 1 / (1 + np.exp(-alpha * x))

# Define the acceleration and deceleration function
def accel_decel(value, target_value, accel_rate=1.0, decel_rate=1.0, time_delta=1):
    # Calculate the difference between the current value and the target value
    diff = target_value - value
    
    # Determine whether to accelerate or decelerate based on the sign of the difference
    if diff > 0:
        rate = accel_rate
    else:
        rate = decel_rate
    
    # Apply the sigmoid function to the difference with the acceleration/deceleration rate and time delta
    smoothed_diff = sigmoid(diff / time_delta, rate)
    
    # Calculate the amount by which to change the value based on the smoothed difference and time delta
    change_amount = smoothed_diff * time_delta
    
    # Calculate the new value based on the change amount
    new_value = value + change_amount
    
    return new_value

# ---------------- Prediction
print('Prediction')

for _ in range(10):
    observation, done = env.reset(), False
    episode_reward = 0.0

    num_envs = 1
    # Episode start signals are used to reset the lstm states
    if args.algo == "ppo_lstm":
        lstm_states = None
        episode_starts = np.ones((num_envs,), dtype=bool)

    num_iter = 0
    value_x = 0.
    value_y = 0.
    time_delta = 0.1

    while not done:
        if args.algo == "ppo_lstm":
            action, lstm_states = model.predict(
                observation,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True)
        else:
            action, _state = model.predict(observation, deterministic=True)

        # Might be useful for stopping the model when reaching the goal
        action = action if np.linalg.norm(action) >= 0.05 else np.array([[0.0, 0.0]])

        action[0][0] = accel_decel(value_x, action[0][0], time_delta=time_delta)
        action[0][1] = accel_decel(value_y, action[0][1], time_delta=time_delta)
        observation, reward, done, info = env.step(action)
        value_x = action[0][0]
        value_y = action[0][1]

        episode_reward += reward
        episode_starts = done
        print(action)
        num_iter += 1
        time.sleep(0.033)
        done = bool(num_iter > 200)
    for i in range(200):
        env.step(np.array([[-0.1,-0.1]]))
        time.sleep(0.033)
    print([episode_reward])

env.close()
