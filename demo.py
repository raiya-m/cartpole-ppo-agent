import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, "videos/", name_prefix="ppo-cartpole")

model = PPO.load("models/ppo_cartpole")

obs, info = env.reset()
terminated, truncated = False, False

while not (terminated or truncated):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

env.close()