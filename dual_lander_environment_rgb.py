import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from gym.envs.box2d.lunar_lander import LunarLander

class DualLunarLanderEnvironment(gym.Env):
    def __init__(self):
        super(DualLunarLanderEnvironment, self).__init__()

        # Use the standard Lunar Lander environment as a base for each lander
        self.lander1 = gym.make("LunarLander-v2", render_mode='rgb_array')
        self.lander2 = gym.make("LunarLander-v2", render_mode='rgb_array')

        # Define the action space (4 actions for each lander)
        self.action_space = spaces.Tuple((self.lander1.action_space, self.lander2.action_space))

        # Define the observation space (combine the observations from both landers)
        lander_obs_space = self.lander1.observation_space.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * lander_obs_space,), dtype=np.float32)

        # Initialize state
        self.state = None

    def step(self, actions):
        # Unpack the actions tuple
        action1, action2 = actions

        # Apply actions to each lander and update environment state
        obs1, reward1, done1, truncated1, info1 = self.lander1.step(action1)
        obs2, reward2, done2, truncated2, info2 = self.lander2.step(action2)

        # Update the environment state
        # self.state = np.concatenate([np.array(obs1), np.array(obs2)])

        # Combine the observations from both landers for full observation
        full_obs1 = np.concatenate([np.array(obs1), np.array(obs2)])
        full_obs2 = np.concatenate([np.array(obs2), np.array(obs1)])

        # Update the environment state with combined observations
        self.state = np.concatenate([full_obs1, full_obs2])

        # Calculate the combined reward
        reward = self._get_reward(np.array(obs1), np.array(obs2), reward1, reward2)

        # Check if the episode is done
        done = done1 or done2

        # Aggregate the info dictionaries (if necessary)
        info = {**info1, **info2}

        return self.state, reward, done, info



    def reset(self):
        # Reset both landers and extract observations
        obs1, _ = self.lander1.reset()
        obs2, _ = self.lander2.reset()

        # Combine the observations from both landers for full observation
        full_obs1 = np.concatenate([np.array(obs1), np.array(obs2)])
        full_obs2 = np.concatenate([np.array(obs2), np.array(obs1)])

        # Reset the state with combined observations
        self.state = np.concatenate([full_obs1, full_obs2])

        return self.state


    def render(self, mode='human'):
        # Render each lander to an RGB array
        img1 = self.lander1.render()
        img2 = self.lander2.render()

        # Combine the images side by side
        combined_img = np.concatenate((img1, img2), axis=1)

        if mode == 'human':
            if not hasattr(self, 'fig'):
                self.fig, self.ax = plt.subplots()
                self.image = self.ax.imshow(combined_img, animated=True)
                plt.axis('off')
                plt.show(block=False)
            else:
                self.image.set_data(combined_img)
                plt.draw()
                plt.pause(0.001)
        elif mode == 'rgb_array':
            return combined_img

    def _get_reward(self, obs1, obs2, reward1, reward2):
        # Basic combined reward from both landers
        combined_reward = reward1 + reward2

        # Extract positions of both landers
        # Assuming the positions are the first two elements in the observation
        x1, y1 = obs1[0], obs1[1]
        x2, y2 = obs2[0], obs2[1]

        # Calculate the Euclidean distance between the two landers
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Define a safe distance threshold
        safe_distance = 0.5  # This can be adjusted based on the scale of your environment

        # If landers are too close, apply a collision penalty
        if distance < safe_distance:
            collision_penalty = 50  # The severity of the penalty can be adjusted
            combined_reward -= collision_penalty

        return combined_reward

# Instantiate and use the environment
env = DualLunarLanderEnvironment()
observation = env.reset()
done = False


while not done:
    # Sample actions for both landers
    action_tuple = env.action_space.sample()

    # Extract individual actions for each lander
    action1, action2 = action_tuple[0], action_tuple[1]

    # Pass the individual actions to the environment's step method
    observation, reward, done, info = env.step((action1, action2))
    
    # Render the combined frame
    env.render()

    if done:
        observation = env.reset()

env.close()

