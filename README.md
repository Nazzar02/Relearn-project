# Group Recommendation with Deep Reinforcement Learning

This repository contains the implementation of a *Group Recommender System* using *Deep Reinforcement Learning (DQN)*. The environment is modeled as an MDP, and the agent learns to recommend movies to groups based on past interactions.

## Files Overview

- **sub_generator.py** - Generates synthetic group data from individual user ratings.
- **env.py** - Defines the reinforcement learning environment, handling group interactions and rewards.
- **agent.py** - Implements the DQN agent that learns to recommend movies to groups.
