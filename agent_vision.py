import numpy as np
import random
from collections import namedtuple, deque

from PixelQNetwork import QNetwork_vision
from model_vision_2 import QNetwork_vision2

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent_vision:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, BUFFER_SIZE = int(1e5), BATCH_SIZE = 64, GAMMA = 0.99, TAU = 1e-3, LR = 5e-4, UPDATE_EVERY = 4, fc1_units=64, fc2_units=64, double_dqn=False, dueling_dqn=False, prioritized_experience_replay=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            BUFFER_SIZE (int): buffer size for Replay buffer
            BATCH_SIZE (int): batch size for Replay buffer
            GAMMA (float): Gamma value for discount factor
            TAU (float): interpolation parameter
            LR (float): learning rate
            UPDATE_EVERY (int): number of times to update table
            fc1_units (int): hidden size of first hidden layer
            fc2_units (int): hidden size of second hidden layer
            double_dqn (bool): whether to use double DQN or not
            dueling_dqn (bool): whether to use dueling or not
            prioritized_experience_replay (bool): whether to use prioritized experience replay or not
        """

        # SETUP params
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR = LR
        self.UPDATE_EVERY = UPDATE_EVERY
        self.DOUBLE_DQN = double_dqn
        self.DUELING = dueling_dqn

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.PER = prioritized_experience_replay

        # Q-Network
        #self.qnetwork_local = QNetwork_vision(state_size, action_size).to(device)
        #self.qnetwork_target = QNetwork_vision(state_size, action_size).to(device)
        self.qnetwork_local = QNetwork_vision(state_size[1], state_size[2], action_size).to(device)
        self.qnetwork_target = QNetwork_vision(state_size[1], state_size[2], action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, prioritized_experience_replay=prioritized_experience_replay)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state[None, :, :], action, reward, next_state[None, :, :], done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        if self.PER: states, actions, rewards, next_states, dones, idx, weights = experiences
        else: states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        oldvals = self.qnetwork_target(next_states).detach()
        if self.DOUBLE_DQN:
            max_indx = torch.argmax(self.qnetwork_local(next_states).detach(), 1).unsqueeze(1)
            Q_targets_next= oldvals.gather(1, max_indx)
        else: Q_targets_next = oldvals.max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.PER:
            weights = torch.from_numpy(weights)
            weights = weights.unsqueeze(1)
            Q_targets *= weights
            Q_expected *= weights

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        if self.PER:
            # update older priorities with new losses distances
            new_priorities = abs(Q_expected - Q_targets)
            self.memory.update_priorities(idx, new_priorities)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# Helped to code Prioritized Experience Replay using tutorial:
# https://davidrpugh.github.io/stochastic-expatriate-descent/pytorch/deep-reinforcement-learning/deep-q-networks/2020/04/14/prioritized-experience-replay.html
# as well followed some parts here:
#   https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, prioritized_experience_replay=False, alpha=0.3):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            prioritized_experience_replay(bool): whether to have Prioritized Experience Replay (PER). or not
            alpha: value to use in (PER) calculation of probabilities
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.PER = prioritized_experience_replay
        self.alpha = alpha
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        if self.PER:
            self.max_priority = 0
            self.priorities = deque(maxlen=buffer_size)
        #     self.buffer_pos = 0
        #     self.prioritized_experience = np.empty(buffer_size, dtype=[("priority", np.float32), ("experience", self.experience)])
        #     self.priorities = np.ones((buffer_size,), dtype=np.float32)

        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        if self.PER:
            prio = 1 if len(self.priorities) < self.batch_size else max(self.priorities)
            self.priorities.append(prio)

        self.memory.append(e)

    def sample(self, beta=0.03):
        """Randomly sample a batch of experiences from memory."""
        if not self.PER:
            experiences = random.sample(self.memory, k=self.batch_size)
        else:
            # X is the number of steps before updating memory
            probs = np.array(self.priorities) ** self.alpha
            probs /= probs.sum()

            length = len(self.memory)
            idxs = np.random.choice(np.arange(length), size=self.batch_size, replace=True, p=probs)
            weights = (length * probs[idxs]) ** (-beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)

            # select the experiences and compute sampling weights
            experiences = [self.memory[i] for i in idxs]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        if not self.PER: return (states, actions, rewards, next_states, dones)
        else: return (states, actions, rewards, next_states, dones, idxs, weights)

    def update_priorities(self, indx, new_prio):
        """Updates priority of experience after learning."""
        for new_p, i in zip(new_prio, indx):
            new_p = new_p.item() ** self.alpha
            self.priorities[int(i)] = new_p
            if new_p > self.max_priority: self.max_priority = new_p

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)