[image1]: https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/images/generated_run.gif?raw=true "Trained Agent"
[optuna1]: https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/images/optuna_detail_runs.png?raw=true "Optuna Detail Run"
[optuna2]: https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/images/optuna_parameters_coordinates.png?raw=true "Optuna Parameters Coordinates"
[optuna3]: https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/images/optuna_parameters_importance.png?raw=true "Optuna Parameters Importance"
[scores]: https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/images/image_run_plotted.png?raw=true "Scores plot - Run "

# AI Agent 🤖 to navigate in a Banana 🍌 world 🌐 [**Report**]

------
Image of player being trained on Vanilla DQN:
![Trained Agent with DQN][image1]

## Introduction

This project aimed to get a sense and discover the potential of DQNs.<br>

Initially this project aimed to use a basic DQN (with pytorch) to solve a problem of an AI agent navigating in a 
world and collecting  bananas (yellow gives points, purple takes away)<br>

The stretch goal was to develop successfully 3 extra types of DQNs:
* [Double DQN](https://arxiv.org/abs/1509.06461)
* [Dueling DQN](https://arxiv.org/abs/1511.06581)
* [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952)

In case sucessfully implementation, there was an optional challenge of 
switching the input of the agent from **state vector** to **pixels** (to simulate computer vision)

## Methodology

I started by taking the notes from the [Udacity class - Reinforced Learning](https://classroom.udacity.com/nanodegrees/nd893/dashboard/overview). <br>
There, I could find and adapt both the Agent, and Model for a Vanilla DQN, as well as the implementation for the Q-Learning.

After successfully running and passing the criteria, I moved into introducing **[Optuna](https://optuna.org)** to try to aid in finding the best parameteres.

Once I had this I started developing the rest of the networks: `Double DQN`, `Dueling DQN` and `Priorized Replay Experience DQN` 

Having successfully passed these, I decided to try to take a chance at solving the pixel version.

## Learning Algorithm

The learning is done by the [Agent](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/agent.py) class,
together with the [Model](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/model.py) class which represents the local and target Network.
In order to ease the training, the agent uses a [ReplayBuffer](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/agent.py#L99) class.

### Agent Details

The agent actions and learning are defined at [agent.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/agent.py).

The learning takes bellow initial parameters that guide the training:
```
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
```

#### Agent functions:

**Step Function** to save each action and respecitve experience (rewards, ...) and learn from that step: <br>
`def step(self, state, action, reward, next_state, done):`

**Act Function** which takes a state and returns accordingly the action as per current policy  <br>

`def act(self, state, eps=0.)`

**Learn Function** which updates accordingly the networks (local and target)  
`def learn(self, experiences, gamma):`

**Soft Update Function** performs a soft update to the model parameters
`def soft_update(self, local_model, target_model, tau)`

#### Agent Auxiliary variables

Besides the functions described above, the Agent also uses a set of Variables/Objects to help its functioning.<br>
Out of which, important to mention **memory** which is an object of [ReplayBuffer](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/agent.py#L99) class
as well as **self.qnetwork_local** and **self.qnetwork_target** which is an object of [QNetwork](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/model.py)

### Model NeuralNetwork

The NeuralNetwork (defined at [model.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/model.py)) is composed by 2 initial Linear Layers. (made them flexbile to receive the hidden sizes via parameter) <br>

Then depending on whether it's selected a Dueling DNQ or not, the network contains
a last Linear Layer for the Output (in case it's not Dueling) or 2 Linear Layers (A and V which will later be combined to get the output value) [more information]([Dueling DQN](https://arxiv.org/abs/1511.06581))   

```
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
if not self.DUELING:
    self.fc3 = nn.Linear(fc2_units, action_size)
else:
    self.fc3_output_v = nn.Linear(fc2_units, action_size)
    self.fc3_output_a = nn.Linear(fc2_units, action_size)
```

The difference between Dueling or not, can also be seen in the forward method, 
where in case of not using Dueling we just return the output processed sequentially through the Linear Layers.
However, in the case of Dueling, the outut is combined using both layers: `fc3_output_v` and `fc3_output_a`.

As per the code:
```
if not self.DUELING:
    """Build a network that maps state -> action values."""
    x = self.fc3(x)
else:
    x_a = self.fc3_output_a(x)
    x = self.fc3_output_v(x) + x_a - x_a.mean(dim=1).unsqueeze(1)
```

### ReplayBuffer 

Buffer used to store experiences which the Agent can learn from.<br>
The Buffer is initialized with the option to have **Prioritizd Experience Replay** of not and adjusted the methods accordingly.

#### ReplayBuffer methods

**Add Function** to add an experience to the buffer <br>
`def add(self, state, action, reward, next_state, done)`

**Sample Function** that takes out a sample of `batch_size` from the buffer.
Depending on whether PER is on or not, it checks the priorities and weights accordingly before sampling.

`def sample(self, beta=0.03)`

**Update Priorities Function** given a set of indexes from the buffer, updates accordingly with the new priorities.
This function is only called in the case of PER (Prioritized Experience Replay) which updates the priorities / weights of the experiences accordingly after going through an experience.

`def update_priorities(self, indx, new_prio)`

### Hyper-Parameters

The project has some parameters which we can tweak to improve the performance. <br>
They are the ones also used with Optuna to try to achieve a best result.

You can find them below, followed by their _search-space_ used by Optuna to find the optimal value.
* `'GAMMA': [0.85, 1],` # discount value used to discount each experiences, thought that going below 0.85 would turn the agent too "stubborn" to learn so made that limit
* `'TAU': [1, 6],` # value for interpolation: **attention** this value is them modified to 1e-TAU, so the real range is between 1e-6, 1e-1
* `'LR': [3, 6],` # learning rate used for optimizer of gradient-descend: **attention** this value is them modified to 1e-TAU, so the real range is between 1e-6, 1e-3
* `'FC1_UNITS': [32,64,128,256],` # Values for 1st Hidden Linear Layer   
* `'FC2_UNITS': [32,64,128,256],`# Values for 1st Hidden Linear Layer
* `'EPS_END': [1, 6],` # lower limit of EPS (used for greedy approach): **attention** this value is them modified to 1e-TAU, so the real range is between 1e-6, 1e-1
* `'EPS_DECAY': [0.85, 1],` # value for which EPS (used for greedy approach) is multiplied accordingly to decrease until reaching the lower limit
* `'USE_DOUBLE_DQN': False / True,` # whether to use a Double DQN network
* `'USE_DUELING_DQN': False / True,` # whether to use a Dueling DQN network
* `'USE_PRIORITIZED_EXP_REP':False / True,`# whether to use a Prioritized Experience Replay DQN network

#### Images from HyperParameters
Below are an example of the Hyper Parameters search from Optuna:
![Optuna Detail Runs][optuna1]
![Optuna Parameters Coordinates][optuna2]
![Optuna Parameters Importance][optuna3]

## Plot of Rewards
Below are an example of how the agent behaves and collects scores after each episode:
![Run of scores][scores]
<br>(The number of episodes needed to solve the game, it's in Jupyter Notebook)

## Ideas for the Future

There is always room for improvement. <br>
From the vast number of ideas, or options that one can do to improve the performance of the agent, 
will try to least some of them to give some food-for-tought.

Ideas for the future:
* Make a more complex Neural Network than the one presented here
* Increase the amount of time for running Optuna HyperParameters search. 
  Unfortunately as the enviroment has consideravel ammount of random aspects to it 
  (eps, the world itself, ...) one can not be entirely sure that the model didn't just "got lucky".
  In order to improve that and take away some uncertainty, one option would be to run it much more times and evaluate better the aspects of each variable tweak.
* One could try to find take a look at implementing other advance techniques which weren't discussed here. (such as: ...)
* 

------

## Rubric / Guiding Evaluation
[Original Rubric](https://review.udacity.com/#!/rubrics/1889/view)

#### Training Code

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ Training Code  |  The repository (or zip file) includes functional, well-documented, and organized code for training the agent. |
| ✅ Framework  |  The code is written in PyTorch and Python 3. |
| ✅ Saved Model Weights  |  The submission includes the saved model weights of the successful agent. |

#### README

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ `README.md`  | The GitHub (or zip file) submission includes a `README.md` file in the root of the repository. |
| ✅  Project Details  | The README describes the the project environment details (i.e., the state and action spaces, and when the environment is considered solved). |
| ✅  Getting Started | The README has instructions for installing dependencies or downloading needed files. |
| ✅  Instructions | The README describes how to run the code in the repository, to train the agent. For additional resources on creating READMEs or using Markdown, see here and here. |

#### Report

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| ✅ Report  | The submission includes a file in the root of the GitHub repository or zip file (one of `Report.md`, `Report.ipynb`, or `Report.pdf`) that provides a description of the implementation. |
| ✅ Learning Algorithm  | The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks. |
| ✅  Plot of Rewards  | A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment. |
| ❗  Ideas for Future Work  | The submission has concrete future ideas for improving the agent's performance. |

#### Bonus :boom:
* ✅ Include a GIF and/or link to a YouTube video of your trained agent!
* ✅ Solve the environment in fewer than 1800 episodes!
* ✅✅✅ Implement a [double DQN](https://arxiv.org/abs/1509.06461), a [dueling DQN](https://arxiv.org/abs/1511.06581), and/or [prioritized experience replay](https://arxiv.org/abs/1511.05952)!
* ❗ For an extra challenge **after passing this project**, try to train an agent from raw pixels! Check out `(Optional) Challenge: Learning from Pixels` in the classroom for more details.
* ❗ Write a blog post explaining the project and your implementation!

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `<root_folder>` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

------
