[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# AI Agent :space_invader: to navigate in a Banana :banana: world :globe_with_meridians:
![Trained Agent][image1]
## Introduction

This project aims to explore the power of teaching an agent through Reinforced Learning (RL) to navigate on a Banana World.

The agents uses a DQN Network with a Deep Q-Learning Aldorithm to learn how to navigate efficiently in the virtual world collecting bananas.

### Options for different Networks

The implementation contains a `simple DQN` (that should solve the game within the first ±260 episodes), 
then there's also a `complex layer DQN` (that should solve the game within the first ±260 episodes) 
and finally you can choose to use a `double DQN` as well (that should solve the game within the first ±260 episodes).

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

3. Place the downloaded file for your environment in the DRLND GitHub repository, in the ``RL-SmartAgent-BananaGame`` folder, and unzip (or decompress) the file. 

## Project Details

### Rules of The Game

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

#### State Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
Given this information, the agent has to learn how to best select actions. 

#### Action Space
Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

#### Conditions to consider solved

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Instructions

### Files

#### Code
1. [agent.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/agent.py) - Agent class containing Q-Learning algorithm
1. [model.py](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/model.py) - Simple DQN model class setup 
1. [Navigation.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation.ipynb) - Jupyter Notebook for running experiment, with simple navigation (getting state space through vector)
1. [Navigation_Pixels.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation_Pixels.ipynb) - Jupyter Notebook for running experiment, with pixel navigation (getting state space through pixeis)

#### Documentation
1. [README.md](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/README.md) - This file
1. [Report.md](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Report.md) - Detailed Report on the project

### Running Normal navigation with state space of `37` dimensions
After fulling the requirements on section [Getting Started](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame#getting-started) and at 
[requirements.txt](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/requirements.txt) 
0. Load Jupyter notebook [Navigation.ipynb](https://github.com/joao-d-oliveira/RL-SmartAgent-BananaGame/blob/main/Navigation.ipynb)
1. Adapt dictionary `SETUP = {` with the desired paramenters
2. Load the environment. Running sections: 
   > 1 Initial Setup <br>
   > 2.1 Start the Environment <br>
   > 2.2. Helper Functions
3. If you want to find other HyperParameters, feel free to run section below and adapt SETUP dictionary accordingly
   > 2.3 Find HyperParameters (Optuna)
4. After your pleased with the parameters setup, run section below and check the ploting of results
   > 2.4 Running full Run with Hyper Parameters 

### Pixel navigation through pixels

