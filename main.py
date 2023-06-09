import gym
import numpy as np

env = gym.make('CartPole-v1', render_mode="human")

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs[0])
        obs = env.step(action)
        env.render()
        episode_rewards += obs[1]
        if obs[2]:
            break
    totals.append(episode_rewards)

env.close()

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
