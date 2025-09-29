import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)

timesteps = 100_000
model.learn(total_timesteps=timesteps)

model.save("models/ppo_cartpole")

env.close()