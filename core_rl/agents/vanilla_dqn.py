# Beginner summary: This file implements a beginner-friendly vanilla DQN agent with epsilon-greedy action selection and Q-learning updates.
# Vanilla DQN Agent
# DQN can be used for discrete action spaces examples: CartPole which has 2 actions (left and right) , LunarLander which has 4 actions (left, right, up, down)

from __future__ import annotations # Type hints without quotes e.g. without this ->  config: 'DQNConfig' and with this -> config: DQNConfig

from dataclasses import dataclass # Automatically gets __init__, __repr__, __eq__, and more
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core_rl.agents.base_agent import BaseAgent
from core_rl.buffers.replay_buffer import ReplayBatch
from core_rl.networks.mlp import MLP

@dataclass(slots=True) # Slots make the class memory efficient
class VanillaDQNConfig:
    """Hyperparameters governing the DQN agent's learning behavior."""

    lr: float = 1e-3              # How big of a step the optimizer takes.
    gamma: float = 0.99           # Discount factor (0.99 means we care deeply about long-term future rewards).
    epsilon_start: float = 1.0    # Start by taking 100% random actions to explore the environment.
    epsilon_decay: float = 0.995  # Multiply epsilon by this number every episode to slowly transition to exploitation.
    epsilon_min: float = 0.01     # Never drop below 1% random actions, ensuring we never completely stop exploring.
    target_update_freq: int = 100 # How often (in steps) we sync the Target Network with the Online Network.
    max_grad_norm: float = 1.0    # Cap for gradient clipping to prevent exploding gradients.


class VanillaDQNAgent(BaseAgent):
    """Vanilla Deep Q-Network agent for discrete action spaces."""

    def __init__(
        self,
        state_dim: int, # Number of features in the state space
        action_dim: int, # Number of possible actions
        config: VanillaDQNConfig, # Configuration object
        device: torch.device # Device to run the agent on (CPU/GPU)
    ):
        self.config = config # Configuration object
        self.device = device # Device to run the agent on (CPU/GPU)
        self.action_dim = action_dim # Number of possible actions
        self.epsilon = self.config.epsilon_start # Current epsilon value
        self.step_count = 0 # Number of steps taken so far

        # -----------------------------------------------------------------
        # NEURAL NETWORK SETUP
        # -----------------------------------------------------------------
        
        # 1. Online Network: The active brain. We calculate gradients and update its weights.
        self.q_network = MLP(state_dim, action_dim).to(self.device)
        
        # 2. Target Network: The stable anchor. We DO NOT train this directly.
        self.target_network = MLP(state_dim, action_dim).to(self.device)
        
        # Initialize both networks with the exact same starting weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Put target network in evaluation mode (disables things like dropout if present)
        self.target_network.eval() 

        # Optimizer only tracks the parameters of the Online Network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.lr)
        
        # Huber Loss (Smooth L1) is preferred over MSE in DQN. 
        # It prevents massive TD errors from completely wrecking the network weights early in training.
        self.loss_fn = nn.SmoothL1Loss()


    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Observes a state and chooses an action. 
        Uses an Epsilon-Greedy strategy to balance exploration and exploitation.
        """
        # EXPLORATION: Flip a weighted coin. If it lands under epsilon, pick a random action.
        # random.random() returns a random float between 0 and 1
        # random.randrange(self.action_dim) returns a random integer between 0 and self.action_dim
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # EXPLOITATION: Ask the neural network for the best action.
        # Neural nets expect a "batch" dimension. So we need to change the shape of the state from [x,y,z] (3,) to [[x,y,z]] (1,3)
        # unsqueeze(0) adds a new dimension at the 0th position - example [x,y,z] -> [[x,y,z]]
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # torch.no_grad() tells PyTorch not to track mathematical operations for gradients,
        # making action selection significantly faster and more memory efficient.
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
        # q_values might look like: [[1.5, 3.2, 0.1]]. 
        # argmax finds the index of the highest number (index 1 here).
        return int(q_values.argmax(dim=1).item())


    def update(self, batch: ReplayBatch) -> dict[str, float]:
        """
        The core learning step. Samples memories and teaches the network the Bellman Equation.
        """
        # Convert raw numpy batches into PyTorch tensors on the correct device (CPU/GPU).
        # unsqueeze(1) adds a new dimension at the 1st position - example [a,b,c] (3) -> [[a],[b],[c]] (3,1)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device) # (batch_size, state_dim)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device).unsqueeze(1) # (batch_size, 1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device).unsqueeze(1) # (batch_size, 1)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device) # (batch_size, state_dim)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device).unsqueeze(1) # (batch_size, 1)

        # -----------------------------------------------------------------
        # STEP 1: CALCULATE CURRENT Q-VALUES (The Agent's Prediction)
        # -----------------------------------------------------------------
        # Pass the batch of states through the network. It outputs Q-values for ALL actions.
        # Example output for 2 actions: [[1.2, 0.5], [0.1, 0.9], ...]
        all_q_values = self.q_network(states)
        
        # We only care about the Q-value for the action the agent ACTUALLY took in that memory.
        # gather() acts like a filter, extracting only the Q-values corresponding to our 'actions' tensor.
        # Example: all_q_values = [[1.2, 0.5], [0.1, 0.9]]
        # actions = [[0], [1]]
        # current_q = [1.2, 0.9]
        current_q = all_q_values.gather(dim=1, index=actions)

        # -----------------------------------------------------------------
        # STEP 2: CALCULATE TARGET Q-VALUES (The Ground Truth / Bellman Target)
        # -----------------------------------------------------------------
        # We use the TARGET network to evaluate the next states to keep learning stable.
        with torch.no_grad():
            # max(dim=1) finds the highest Q-value for each next state across all actions
            # Returns (max_values, max_indices) - we only want the values [0]
            # keepdim=True maintains the column shape for broadcasting with rewards
            # Example: [[1.2, 0.5], [0.1, 0.9]] -> max(dim=1) -> (tensor([[1.2], [0.9]]), tensor([[0], [1]]))
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            
            # The Bellman Equation: Reward + (Gamma * Next State Value)
            # If the episode ended (done=1), the future value is 0. 
            # (1.0 - dones) handles this masking automatically.
            target_q = rewards + self.config.gamma * next_q_values * (1.0 - dones)

        # -----------------------------------------------------------------
        # STEP 3: CALCULATE LOSS AND UPDATE WEIGHTS
        # -----------------------------------------------------------------
        # Measure how far off our prediction was from the Bellman target.
        loss = self.loss_fn(current_q, target_q)
        
        # Standard PyTorch gradient update sequence
        self.optimizer.zero_grad()  # Clear old gradients
        loss.backward()             # Calculate new gradients based on the loss
        
        # Gradient Clipping: Prevents sudden, massive weight changes that can ruin the network.
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()       # Nudge the weights in the correct direction

        # -----------------------------------------------------------------
        # STEP 4: TARGET NETWORK SYNC
        # -----------------------------------------------------------------
        self.step_count += 1
        # Every N steps, drag the target network's weights up to match the online network.
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": float(loss.item())}


    def decay_epsilon(self) -> None:
        """Called at the end of every episode to reduce the exploration rate."""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)