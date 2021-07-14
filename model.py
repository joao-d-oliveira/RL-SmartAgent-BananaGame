import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, dueling_dqn=False, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            dueling_dqn (bool): Turn on or off Dueling DQN
            fc1_units (int): 1st hidden layer size
            fc2_units (int): 2nd hidden layer size
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.DUELING = dueling_dqn

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        if not self.DUELING:
            self.fc3 = nn.Linear(fc2_units, action_size)
        else:
            self.fc3_output_v = nn.Linear(fc2_units, action_size)
            self.fc3_output_a = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        if not self.DUELING:
            """Build a network that maps state -> action values."""
            x = self.fc3(x)
        else:
            x_a = self.fc3_output_a(x)
            x = self.fc3_output_v(x) + x_a - x_a.mean(dim=1).unsqueeze(1)

        return x
