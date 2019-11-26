[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
# Navigation
-----------------------------------

### The Environment

![Trained Agent][image1]

The environment is a square world where the agent is trained to collect bananas -  the yellow ones while avoiding the blue ones. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

  - `0`: move forward.
  - `1`: move backward.
  - `2`: turn left.
  - `3`: turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Installing Dependencies
1. Clone the repository:
`
git clone https://github.com/rehamelkholy/banana-navigation.git
`
2. Follow the instructions [here ](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install the dependencies. By successfuly following the instructions you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

__*(For Windows users)*__ The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### Setting up the Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Unzip the downloaded UnityEnvironment to the folder you will be working.

3. Makse sure you have downloaded and installed Anaconda. You can download it from https://www.anaconda.com/distribution/

4. Now you can create your environment. Since this environment refers to a Udacity project for the Deep Reinforcement Learning Nanodegree, lets call our environment `DRLND`.

    Linux or Mac:
    ```sh
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    
    Windows:
    ```sh
    conda create --name drlnd python=3.6 
    activate drlnd
    ```
    
    
5. We will be working with pytorch version 0.4.0 (an early version), so make sure that you install this version of pytorch first by typing:
    ```sh
    conda install pytorch=0.4.0 -c pytorch
    ```
    
6. Perform a minimal installation of the OpenAI Gym environment (see instructions here: https://github.com/openai/gym)

7. For the rest of the prerequisities please do type:
    ```sh
    pip install .
     ```
     The above line of code assumes that at the folder you are working, you have the `setup.py` which includes the UnityAgents and the `requirements.txt` file that contains other useful packages (that exist in that repository).
     
8. Create a Python execution backend for Jupyter for the drlnd environment
    ```sh
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
     
Now you are not only ready to use the UnityAgents evnironment, but the OpenAI Gym as well.  You are all set to start playing with reinforcement learning environments! Yay!

Other useful utilities will also be installed if you follow these directions, including Jupyter Notebook, so consider the above installation guide as a complete guide to setup your RL environments!

### Running the Code

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### Report

To get a deeper understanding of the working of the code, there is a `Report.pdf` file which aids in understanding *under-the-hood* working of the agent and the underlying model.