import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


from gym_chrono.envs.wheeled.rassor_traction import rassor_traction

if __name__ == '__main__':
    env = rassor_traction()
    check_env(env)

    obs, _ = env.reset()
    env.render()

    print(env.observation_space)

    # Define the ranges
    range_1_low, range_1_high = -np.pi/2, np.pi/2
    range_2_low, range_2_high = -0.25, 0.25

    # Generate random actions for each part
    random_actions_1 = np.random.uniform(range_1_low, range_1_high, 6)
    random_actions_2 = np.random.uniform(range_2_low, range_2_high, 2)

    # Combine the two parts
    random_action = np.concatenate((random_actions_1, random_actions_2))

    print(random_action)


    n_steps = 1000000
    for step in range(n_steps):
        print(f"Step {step + 1}")
        obs, reward, terminated, truncated, info = env.step(random_action)
        done = terminated or truncated
        print("obs=", obs, "reward=", reward, "done=", done)
        env.render()
        if done:
            print("reward=", reward)
            break
