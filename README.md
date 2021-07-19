[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# AI Agent 🤖 to navigate in a Banana 🍌 world 🌐
![Trained Agent][image1]

## Introduction

This project aims to explore the power of teaching an agent through Reinforced Learning (RL) to navigate on a Banana World.

The agents uses a DQN Network with a Deep Q-Learning Aldorithm to learn how to navigate efficiently in the virtual world collecting bananas.

## Options for different Networks

The implementation options for: `Vanilla DQN`, `Double DQN`, `Dueling DQN` and `Priorized Replay Experience DQN`.<br>
Please check under [Instructions](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame#instructions) on how to activate each of this options

------
## Getting Started

1. You need to have installed the requirements (specially mlagents==0.4.0).
   Due to deprecated libraries, I've included a [python folder](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/tree/main/python) which will help
   with installation of the system.
      - Clone the repository: `git clone https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame.git`
      - Go to python folder: `cd RL-SmartAgent-BananaGame/python`
      - Compile and install needed libraries `pip install .`
2. Download the environment from one of the links below
   Download only the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS or Collab_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2.1 In case you prefer to test the Visual Environment (where the states are defined by the video instead of a vector that indicates the state).
please Download instead bellow:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)


3. Place the downloaded file for your environment in the DRLND GitHub repository, in the ``RL-SmartAgent-BananaGame`` folder, and unzip (or decompress) the file. 

## Project Details

### Rules of The Game

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

### State Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
Given this information, the agent has to learn how to best select actions.

The state space of the **"Visual"** environment is composed by the snapshot of the video of the game, meaning that is 
an array composed by (84, 84, 3) which means, 84 of width and height and 3 channels (R.G.B.).

### Action Space
Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Conditions to consider solved

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Instructions

### Files

#### Code
1. [agent.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/agent.py) - Agent class containing Q-Learning algorithm and all supoprt for `Vanilla DQN`, `Double DQN`, `Dueling DQN` and `Priorized Replay Experience DQN`.
1. [model.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/model.py) - DQN model class setup (containing configuration for `Dueling DQN`) 
1. [Navigation.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation.ipynb) - Jupyter Notebook for running experiment, with simple navigation (getting state space through vector)
---
1. [agent_vision.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/agent_vision.py) - Agent class containing Q-Learning algorithm Visual environment
1. [model_vision.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/model_vision.py) - DQN model class setup for Visual environment 
1. [Navigation_Pixels.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation_Pixels.ipynb) - Jupyter Notebook for running experiment, with pixel navigation (getting state space through pixeis)

#### Documentation
1. [README.md](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/README.md) - This file
1. [Report.md](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Report.md) - Detailed Report on the project

#### Models
All models are saved on the subfolder (models).
For example, [checkpoint.pt](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/models/checkpoint.pt) is 
a file which has been saved upon success of achieving the goal, and [model.pt](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/models/model.pt) is the end model after runing all episodes.

### Running Normal navigation with state space of `37` dimensions

#### Structure of Notebook

The structure of the notebook follows the following:
> 1. Initial Setup: _(setup for parameters of experience, check [report](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Report.md) for more details)_ <br>
> 2. Navigation <br>
> 2.1 Start the Environment: _(load environment for the game)_<br>
> 2.2 HelperFunctions: _(functions to help the experience, such as Optuna, DQNsearch, ...)_<br>
> 2.3 Baseline DQN: _(section to train an agent with the standard parameters, without searching for hyper-parameters)_<br>
> 2.4 Vanilla DQN: _(section to train an agent with a Vanilla DQN)_<br>
> 2.5 Double DQN: _(section to train an agent with a Double DQN)_<br>
> 2.6 Dueling DQN: _(section to train an agent with a Dueling DQN)_<br>
> 2.7 Prioritized Experience Replay (PER) DQN: _(section to train an agent with a PER DQN)_<br>
> 2.8 Double DQN with PER: _(section to train an agent with a PER and Double DQN at same time)_<br>
> 2.9 Double with Dueling and PER DQN: _(section to train an agent with a PER and Double and dueling DQN)_<br>
> 3.0 Plot all results: _(section where all the results from above sections are plotted to compare performance)_

Each of the sections: [`2.3 Baseline DQN`, `2.4 Vanilla DQN`, `2.5 Double DQN`, `2.6 Dueling DQN`, `2.7 Prioritized Replay DQN`, `2.8 Double DQN with PER`, `2.9 Double with Dueling and PER DQN`]

Have subsessions:
> 2.x.1 Find HyperParameters (Optuna) <br>
> 2.x.1.1 Ploting Optuna Results <br>
> 2.x.2 Run (network) DQN <br>
> 2.x.3 Plot Scores <br>

Each section relevant to the respective DQN. <br>
You can choose whether to use the regular parameters, or try to find them through Optuna

#### Running

After fulling the requirements on section [Getting Started](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame#getting-started) and at 
[requirements.txt](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/requirements.txt) 
0. Load Jupyter notebook [Navigation.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation.ipynb)
1. Adapt dictionary `SETUP = {` with the desired paramenters
2. Load the environment. Running sections: 
   > 1 Initial Setup <br>
   > 2.1 Start the Environment <br>
   > 2.2. Helper Functions
3. Then go the section of the Network you want to run [`2.3 Baseline DQN`, `2.4 Vanilla DQN`, `2.5 Double DQN`, `2.6 Dueling DQN`, `2.7 Prioritized Replay DQN`, `2.8 Double DQN with PER`, `2.9 Double with Dueling and PER DQN`]
   There you will be able to either run Optuna to find the theoretically best parameters, or run the model with the base paramenters.
