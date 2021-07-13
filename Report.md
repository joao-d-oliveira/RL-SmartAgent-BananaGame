[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# AI Agent :space_invader: to navigate in a Banana :banana: world :globe_with_meridians: [**Report**]

------

## Introduction

This is the 

## Methodology

Algorithm with DQN to train an Agent to navigate and collect bananas in a virtual World.

## Learning Algorithm

### Model NeuralNetwork

### Hyperparameters

## Plot of Rewards

## Ideas for the Future

------

## Rubric / Guiding Evaluation
[Original Rubric](https://review.udacity.com/#!/rubrics/1889/view)

#### Training Code

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| :white_check_mark: Training Code  |  The repository (or zip file) includes functional, well-documented, and organized code for training the agent. |
| :white_check_mark: Framework  |  The code is written in PyTorch and Python 3. |
| :white_check_mark: Saved Model Weights  |  The submission includes the saved model weights of the successful agent. |

#### README

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| :white_check_mark: `README.md`  | The GitHub (or zip file) submission includes a `README.md` file in the root of the repository. |
| :exclamation:  Project Details  | The README describes the the project environment details (i.e., the state and action spaces, and when the environment is considered solved). |
| :exclamation:  Getting Started | The README has instructions for installing dependencies or downloading needed files. |
| :exclamation:  Instructions | The README describes how to run the code in the repository, to train the agent. For additional resources on creating READMEs or using Markdown, see here and here. |

#### Report

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------| 
| :exclamation: Report  | The submission includes a file in the root of the GitHub repository or zip file (one of `Report.md`, `Report.ipynb`, or `Report.pdf`) that provides a description of the implementation. |
| :exclamation:  Learning Algorithm  | The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks. |
| :exclamation:  Plot of Rewards  | A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment. |
| :exclamation:  Ideas for Future Work  | The submission has concrete future ideas for improving the agent's performance. |

#### Bonus :boom:
* :exclamation: Include a GIF and/or link to a YouTube video of your trained agent!
* :white_check_mark: Solve the environment in fewer than 1800 episodes!
* :exclamation: Write a blog post explaining the project and your implementation!
* :exclamation: Implement a [double DQN](https://arxiv.org/abs/1509.06461), a [dueling DQN](https://arxiv.org/abs/1511.06581), and/or [prioritized experience replay](https://arxiv.org/abs/1511.05952)!
* :exclamation: For an extra challenge **after passing this project**, try to train an agent from raw pixels! Check out `(Optional) Challenge: Learning from Pixels` in the classroom for more details.

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
