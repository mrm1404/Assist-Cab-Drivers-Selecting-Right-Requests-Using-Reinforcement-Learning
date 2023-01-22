# Assist-Cab-Drivers-Selecting-Right-Requests-Using-Reinforcement-Learning

### Business Problem:

At SuperCabs, a leading app-based cab provider in a large Indian metro city. In this highly competitive industry, retention of good cab drivers is a crucial business driver, and you believe that a sound RL-based system for assisting cab drivers can potentially retain and attract new cab drivers. 

Cab drivers, like most people, are incentivised by a healthy growth in income. The goal of your project is to build an RL-based algorithm which can help cab drivers maximise their profits by improving their decision-making process on the field.

### The Need for Choosing the "Right" Requests
Most drivers get a healthy number of ride requests from customers throughout the day. But with the recent hikes in electricity prices (all cabs are electric), many drivers complain that although their revenues are gradually increasing, their profits are almost flat. Thus, it is important that drivers choose the 'right' rides, i.e. choose the rides which are likely to maximise the total profit earned by the driver that day. 

### Solution:
Taking long-term profit as the goal, we propose a method based on reinforcement learning to optimize taxi driving strategies for profit maximization. This optimisation problem is formulated as a Markov Decision Process.

In this project, we will create the environment and an RL agent that learns to choose the best request. We train agent using vanilla Deep Q-learning (DQN) as demonstrated below.
![Q_network2+-+Architecture+2](https://user-images.githubusercontent.com/39112641/213916651-8f5a875b-ee6d-4636-8304-264b7569ac5e.png)

### Demonstation:
https://github.com/mrm1404/Assist-Cab-Drivers-Selecting-Right-Requests-Using-Reinforcement-Learning/blob/main/DQN_Agent_Arch2.ipynb
