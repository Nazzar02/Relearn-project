####################################
# Agent for the recommender system #
####################################



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import env 
import pandas as pd



#----------------------------------------------------------------------------#
#                           GroupGenerator Class                             #
#----------------------------------------------------------------------------#


class DQNAgent:

    # class constructor

    def __init__(self, state_size, action_size, hidden_size=1024, lr=3e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Replay buffer
        self.memory = deque(maxlen=50000)  # Increased replay buffer size

        # Neural networks for Q-learning (chatGPT helped me there)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    # class methods
    # Build the neural network model
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )

    # Update the target model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Choose an action using epsilon-greedy policy, limited to valid actions
    def act(self, state, valid_actions):
        if len(valid_actions) == 0:
            return None  # No valid actions available
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions)  # Random action from valid actions

        # Ensure state has correct size
        state = np.array(state)
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)), mode='constant')
        elif len(state) > self.state_size:
            state = state[:self.state_size]

        # Choose action with highest Q-value
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        valid_actions = [int(a) for a in valid_actions if a < self.action_size]  # Ensure valid actions are within bounds
        q_values_filtered = {a: q_values[0, a].item() for a in valid_actions}
        return max(q_values_filtered, key=q_values_filtered.get)  # Action with highest Q-value

    # Store experience in replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Train the model using random samples from memory
    def replay(self, batch_size=128):  # Increased batch size
        if len(self.memory) < batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors (chatGPT helped me there)
        states = torch.FloatTensor(np.array([np.pad(s, (0, self.state_size - len(s)), mode='constant') if len(s) < self.state_size else s[:self.state_size] for s in states]))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array([np.pad(ns, (0, self.state_size - len(ns)), mode='constant') if len(ns) < self.state_size else ns[:self.state_size] for ns in next_states]))
        dones = torch.FloatTensor(dones)

        # Current Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss and optimization
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Decay epsilon for exploration-exploitation tradeoff
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


#----------------------------------------------------------------------------#
#                                 main                                       #
#----------------------------------------------------------------------------#

if __name__ == "__main__":


    # load the data
    data_path = "/home/enzoc/RL/ml-latest-small-processed/group_ratings_train.csv"
    data = pd.read_csv(data_path)

    # Init the environment
    num_groups = data["groupId"].nunique()
    num_movies = data["movieId"].nunique()
    environment = env.Env(train_data=data, num_groups=num_groups, num_items=num_movies)

    # Init the agent
    state_size = 20  # increase state size to include more history
    action_size = num_movies
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    # parameters for training, if improved performance is needed, these can be adjusted but computational resources will be required
    num_episodes = 1500  
    max_steps = 300  
    batch_size = 128  

    # rewards per episode
    rewards_per_episode = []

    # baseline random rewards per episode (compare with DQN)
    random_rewards = []

    # loop for training episodes
    for episode in tqdm(range(num_episodes), desc="Entraînement", unit="episode"):
        group_id = np.random.randint(1, num_groups + 1) 
        state = environment.reset(group_id)
        total_reward = 0
        random_total_reward = 0

        for step in range(max_steps):
            valid_actions = data[data["groupId"] == group_id]["movieId"].unique()
            valid_actions = valid_actions[valid_actions < num_movies] 

            # DQN Agent
            action = agent.act(state, valid_actions=valid_actions) 
            if action is None:
                break

            next_state, reward, done, _ = environment.step(group_id, action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Random Baseline
            random_action = np.random.choice(valid_actions) if len(valid_actions) > 0 else None
            if random_action is not None:
                _, random_reward, _, _ = environment.step(group_id, random_action)
                random_total_reward += random_reward

            if done:
                break

        # train the agent
        agent.replay(batch_size=batch_size)
        agent.decay_epsilon()

        rewards_per_episode.append(total_reward)
        random_rewards.append(random_total_reward)

    # show the rewards per episode
    avg_rewards = [np.mean(rewards_per_episode[i:i + 50]) for i in range(0, len(rewards_per_episode), 50)]
    random_avg_rewards = [np.mean(random_rewards[i:i + 50]) for i in range(0, len(random_rewards), 50)]

    for idx, (avg, rand_avg) in enumerate(zip(avg_rewards, random_avg_rewards)):
        print(f"Episodes {idx * 50 + 1}-{(idx + 1) * 50}: Average Reward: {avg:.2f} (DQN), {rand_avg:.2f} (Random)")

    # folder for plots
    plots_dir = "/home/enzoc/RL/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # plot the rewards per episode (chatGPT helped me there)
    plt.figure()
    plt.plot(rewards_per_episode, label="DQN Agent")
    plt.plot(random_rewards, label="Random Baseline")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Rewards per Episode")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "rewards_per_episode_comparison.png"))

    plt.figure()
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards, label="DQN Agent")
    plt.plot(range(1, len(random_avg_rewards) + 1), random_avg_rewards, label="Random Baseline")
    plt.xlabel("50-Episode Intervals")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per 50 Episodes")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "average_rewards_comparison.png"))

    # Validation
    validation_rewards = []
    for _ in range(100):  # 100 episodes for validation
        group_id = np.random.randint(1, num_groups + 1)
        state = environment.reset(group_id)
        total_reward = 0

        for step in range(max_steps):
            valid_actions = data[data["groupId"] == group_id]["movieId"].unique()
            valid_actions = valid_actions[valid_actions < num_movies]
            action = agent.act(state, valid_actions=valid_actions)
            if action is None:
                break

            next_state, reward, done, _ = environment.step(group_id, action)
            state = next_state
            total_reward += reward

            if done:
                break

        validation_rewards.append(total_reward)

    avg_validation_reward = np.mean(validation_rewards)
    print(f"Average Validation Reward: {avg_validation_reward:.2f}")

    # save the model
    agent.update_target_model()
    torch.save(agent.model.state_dict(), "dqn_model.pth")
    print("Modèle entraîné, graphiques sauvegardés et modèle enregistré.")
