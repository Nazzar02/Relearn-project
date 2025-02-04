##########################################
# Environment for the recommender system #
##########################################

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import pandas as pd


#----------------------------------------------------------------------------#
#                           GroupGenerator Class                             #
#----------------------------------------------------------------------------#


class Env:

    # class constructor

    def __init__(self, train_data, num_groups, num_items, history_length=5):
        self.train_data = train_data
        self.num_groups = num_groups
        self.num_items = num_items
        self.history_length = history_length

        # Initialize group history
        self.group_histories = {group_id: [] for group_id in range(1, num_groups + 1)}

    # class methods

    #reset the envorinment for a specific group
    def reset(self, group_id):
        if group_id not in self.group_histories:
            raise ValueError(f"Group {group_id} does not exist.")
        self.group_histories[group_id] = []
        return self._get_state(group_id)

    #simulate the environment's response to an action
    def step(self, group_id, action):

        if group_id not in self.group_histories:
            raise ValueError(f"Group {group_id} does not exist.")

        # Compute reward
        reward = self._compute_reward(group_id, action)

        # Update history
        history = self.group_histories[group_id]
        history.append(action)
        if len(history) > self.history_length:
            history.pop(0)

        next_state = self._get_state(group_id)
        done = False  # No terminal condition in this simplified environment
        info = {}

        return next_state, reward, done, info
    
    #Get the current state for a group
    def _get_state(self, group_id):
        return self.group_histories[group_id]

    #Compute the reward for recommending an item to a group
    def _compute_reward(self, group_id, action):
        group_data = self.train_data[self.train_data['groupId'] == group_id]
        liked_movies = group_data[group_data['rating'] == 1]['movieId'].tolist()
        #print(f"Group {group_id} liked movies: {liked_movies}")  # Debugging
        #print(f"Action: {action}")  # Debugging
        reward = 1 if action in liked_movies else 0
        #print(f"Reward: {reward}")  # Debugging
        return reward


#----------------------------------------------------------------------------#
#                                 main                                       #
#----------------------------------------------------------------------------#


if __name__ == "__main__":

    # load the data
    train_data_path = "/home/enzoc/RL/ml-latest-small-processed/group_ratings_train.csv"
    train_data = pd.read_csv(train_data_path)

    # number of groups and items
    num_groups = train_data['groupId'].max()
    num_items = train_data['movieId'].nunique()

    # Create the environment    
    env = Env(train_data, num_groups, num_items)

    # Test the environment
    group_id = 10
    state = env.reset(group_id)
    print(f"Initial state for group {group_id}: {state}")

    # Siimulation for 10 steps
    for i in range(10):
        # filter the data for the group
        group_data = train_data[train_data['groupId'] == group_id]
        noted_movies = group_data['movieId'].tolist()

        # Chose an action (movieId) for the group
        if noted_movies:
            action = np.random.choice(noted_movies)
        else:
            print(f"Group {group_id} has no noted movies.")
            break  # end if there are no noted movies

        # do the action and get the results
        next_state, reward, done, info = env.step(group_id, action)

        # showing the results (chatGPT helpde me do it nicely)
        
        print(f"Step {i+1}:")
        print(f"  Action (movieId): {action}")
        print(f"  Reward: {reward}")
        print(f"  Next state: {next_state}")
        print("---")
