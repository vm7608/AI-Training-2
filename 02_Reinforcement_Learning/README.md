# **Reinforcement Learning**

## **1. What is Reinforcement Learning?**

- Reinforcement learning is a machine learning training method based on rewarding desired behaviors and punishing undesired ones.

- In general, a reinforcement learning agent -- the entity being trained -- is able to perceive and interpret its environment, take actions and learn through trial and error.

- Key Features of Reinforcement Learning:
  - The agent is not instructed about the environment and what actions need to be taken. It takes the next action and changes states according to the feedback of the previous action.
  - RL based on the trial and error process.
  - Feedback is always delayed, not instantaneous.
  - The environment is stochastic, and the agent needs to explore it to reach to get the maximum positive rewards.

## **2. Terms used in Reinforcement Learning**

- **Agent**: An entity that can perceive/explore the environment and act upon it.

- **Environment (E)**: A scenario that an agent is present or surrounded by. In RL, we assume the stochastic environment, which means it is random in nature.

- **Action (A)**: the set of all possible moves that an agent can take in a given situation within the environment.

- **State (S)** is a situation returned by the environment after each action taken by the agent.

- **Reward (R)**: A feedback returned to the agent from the environment to evaluate the action of the agent.

- **Policy (π)**: It is a strategy which applies by the agent to decide the next action based on the current state.

- **Value (V)**:  It is expected long-term retuned with the discount factor.

- **Q value or action value (Q)**: It is mostly similar to the value, but it takes an extra parameter, which is the current action.

<p align="center">
  <img src="https://www.guru99.com/images/1/082319_0514_Reinforceme1.png" >
  <br>
  <i>Typical RL Scenario</i>
</p>

## **3. Approaches to implement Reinforcement Learning**

### **3.1 Value-Based**

- The value-based approach is about to find the optimal value function, which is the maximum value at a state under any policy. Therefore, the agent expects the long-term return at any state(s) under policy π.

### **3.2 Policy-based**

- In a policy-based RL method, you try to come up with such a policy that the action performed in every state helps you to gain maximum reward in the future.
- Two types of policy-based methods are:
  - Deterministic: For any state, the same action is produced by the policy π.
  - Stochastic: Every action has a certain probability, which is determined by the following equation.

  ```math
  π(a|s) = P[A=a|S=s]
  ```

### **3.3 Model-Based**

- In this Reinforcement Learning method, you need to create a virtual model for each environment. The agent learns to perform in that specific environment.

## **4. Elements of Reinforcement Learning**

- There are four main elements of Reinforcement Learning:
  - Policy
  - Reward function
  - Value function
  - Model

### **4.1 Policy**

- A policy can be defined as a way how an agent behaves at a given time.
  - It maps the perceived states of the environment to the actions taken on those states.
  - A policy is the core element of the RL as it defines the behavior of the agent. In some cases, it may be a simple function or a lookup table which involve general computation as a search process.
  
- It could be deterministic or a stochastic policy:
  - For deterministic policy:

    ```math
    a = π(s)
    ```

  - For stochastic policy:

    ```math
    π(a|s) = P[A=a|S=s]
    ```

### **4.2 Reward function**

- The goal of RL is defined by the reward signal:
  - At each state, the environment sends an immediate signal to the learning agent, and this signal is known as a reward signal.
  - These rewards are given according to the good and bad actions taken by the agent.

- The agent's main objective is to maximize the total number of rewards for good actions.

- The reward signal can change the policy, such as if an action selected by the agent leads to low reward, then the policy may change to select other actions in the future.

### **4.3 Value function**

- The value function gives information about how good the situation and action are and how much reward an agent can expect.

- A reward indicates the immediate signal for each good and bad action, whereas a value function specifies the good state and action for the future.

- The value function depends on the reward as, without reward, there could be no value. The goal of estimating values is to achieve more rewards.

### **4.4 Model**

- The models mimics the behavior of the environment. With the help of the model, one can make inferences about how the environment will behave. Such as, if a state and an action are given, then a model can predict the next state and reward.

- The model is used for planning, which means it provides a way to take a course of action by considering all future situations before actually experiencing those situations.

- The approaches for solving the RL problems with the help of the model are termed as the model-based approach. Comparatively, an approach without using a model is called a model-free approach.

## **5. Bellman Equation**

### **5.1 Example**

- Consider the following problems:

<p align="center">
  <img src="https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-works.png" >
  <br>
  <i>Example problem</i>
</p>

- In the above problem, the agent is at s9 and it have to get to s4 for the diamond.
  - The agent can move in four directions: up, down, left, and right.
  - If it reach s4 (diamond), then it will get a reward of +1.
  - If it reach s8 (fire), then it will get a reward of -1.
  - The agent can take any path to reach the diamond, but it needs to make it in minimum steps.

- Initally, the agent will explore the enviroment and try to reach the diamond. As soon as it reachs the diamond, it will backtrace its step back and mark values of all states which leads towards the goal as V = 1.

<p align="center">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20210914203249/Env11.png" >
  <br>
  <i>Result without Bellman</i>
</p>

- But, if we change the start position, the agent can not find the path to the goal. So, we need to use the Bellman equation to solve this problem.

<p align="center">
  <img src="https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-works3.png" >
  <br>
  <i>Problem without Bellman</i>
</p>

### **5.2 The Bellman Equation**

- The Bellman equation is used to calculate the utility of the states. The utility of the state is the expected long-term return of the state, also known as the value of the state.

- The Bellman equation is given below:

  ```math
  V(s) = max[R(s) + γV(s')]
  ```

- The key-elements used in Bellman equation are:
  - The agent perform an action "a" in the state "s" and moves to the next state "s'".
  - The agent receives a reward "R(s)".
  - V(s): It is the value of the state s.
  - V(s'): It is the value of the next state.
  - γ: It is the discount factor, which determines how much the agent cares about rewards in the distant future relative to those in the immediate future. It has a value between 0 and 1. Lower value encourages short–term rewards while higher value promises long-term reward

- With the above example, we start from block s3 (next to the target). Assume discount factor γ = 0.9.

  - For s3 block. Here V(s') = 0 because no further state to move. So, the Bellman equation will be:

    ```math
    V(s3) = max[R(s3) + γV(s')] = max[1 + 0.9 * 0] = 1
    ```

  - For s2 block. Here V(s') = 1 because it can move to s3. So, the Bellman equation will be:

    ```math
    V(s2) = max[R(s2) + γV(s')] = max[0 + 0.9 * 1] = 0.9
    ```

  - For s1 block. Here V(s') = 0.9 because it can move to s2. So, the Bellman equation will be:

    ```math
    V(s1) = max[R(s1) + γV(s')] = max[0 + 0.9 * 0.9] = 0.81
    ```
  
  - For s5 block. Here V(s') = 0.81 because it can move to s1. So, the Bellman equation will be:

    ```math
    V(s5) = max[R(s5) + γV(s')] = max[0 + 0.9 * 0.81] = 0.73
    ```

  - For s9 block. Here V(s') = 0.73 because it can move to s5. So, the Bellman equation will be:

    ```math
    V(s9) = max[R(s9) + γV(s')] = max[0 + 0.9 * 0.73] = 0.66
    ```
  
  - s9 block is also the start block of agent.

  <p align="center">
    <img src="https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-bellman-equation.png" >
    <br>
    <i>Result using Bellman equation</i>
  </p>

  - Now we move to s7 block, here the agent have 3 option:
    - UP to s3
    - LEFT to s8 (fire)
    - DOWN to s11
    - Can not move left to s6 because it is a wall.

  <p align="center">
    <img src="https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-bellman-equation2.png" >
    <br>
    <i>s7 block</i>
  </p>

  - The `max` in Bellman equation denote the most optimal path among all posible action that agent can take at a given state. Among all these actions available the maximum value for that state is the UP action. So, the Bellman equation will be:

    ```math
    V(s7) = max[R(s7) + γV(s')] = max[0 + 0.9 * 1] = 0.9
    ```

  - Continue the process, we got the final result. The agent will take the path with maximum value by following the increasing value of the states based on the Bellman equation.

<p align="center">
  <img src="https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-bellman-equation3.png" >
  <br>
  <i>Final result</i>
</p>

## **6. Types of Reinforcement Learning**

There are two types of reinforcement learning methods.

### **6.1 Positive**

- The positive reinforcement learning means adding something to increase the tendency that expected behavior would occur again. It impacts positively on the behavior of the agent and increases the strength of the behavior.

- This type of Reinforcement helps you to maximize performance and sustain change for a more extended period. However, too much Reinforcement may lead to over-optimization of state, which can affect the results.

### **6.2 Negative**

- The negative reinforcement learning is opposite to the positive reinforcement as it increases the tendency that the specific behavior will occur again by avoiding the negative condition.

- It can be more effective than the positive reinforcement depending on situation and behavior, but it provides reinforcement only to meet minimum behavior.

## **7. Markov Decision Process**

### **7.1 What is Markov Decision Process?**

- Markov decision process (MDP) is a control process that provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker.

- MDP is used to describe the environment for the RL, and almost all the RL problem can be formalized using MDP.

- MDP contains a tuple of four elements (S, A, Pa, Ra):
  - A set of finite states (S)
  - A set of finite actions (A)
  - A set of rewards (Ra) received after transitioning from state s to s' with action a
  - A transition probability matrix (Pa) which is the probability of transitioning from state s to s' with action a

<p align="center">
  <img src="https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-markov-decision-process.png" >
  <br>
  <i>Markov Decision Process</i>
</p>

### **7.2 Markov Property**

- Markov Property define that "If the agent is present in the current state S1, performs an action a1 and move to the state s2, then the state transition from s1 to s2 only depends on the current state and future action and states do not depend on past actions, rewards, or states."

- In other words, the next state depends only on the current state and action, not on the sequence of events that preceded it. Hence, MDP is an RL problem that satisfies the Markov property.

- Mathematically, the Markov property is defined as:

  ```math
  P[S(t+1) | S(t)] = P[S(t+1) | S(t), S(t-1), S(t-2), ... , S(0)] = P[S(t+1) | S(t)]
  ```

- For example in a Chess game, the players only focus on the current state and do not need to remember past actions or states.

- The Markov property allows MDP problems to be fully characterized by the transition function between states rather than the full history which means that at any point, we have all the information we need to make decisions based just on the current state. The past is irrelevant for determining the next action.

## **8. Reinforcement Learning Algorithms**

### **8.1 Q-Learning**

#### **8.1.1 What is Q-Learning?**

- Q-learning is a model-free reinforcement learning technique used by agents to learn what actions to take under what circumstances. It learns the value function Q (S, a), which means how good to take action "a" at a particular state "s."

- The below flowchart explains the working of Q-learning:

<p align="center">
  <img src="https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-algorithms.png" >
  <br>
  <i>Q-Learning flowchart</i>
</p>

#### **8.1.2 Describe a example**

- Let’s say that a robot has to cross a maze and reach the end point. There are mines, and the robot can only move one tile at a time. If the robot steps onto a mine, the robot is dead. The robot has to reach the end point in the shortest time possible.

- The scoring/reward system is as below:
  - The robot loses 1 point at each step. So that the robot have to take the shortest path and reaches the goal as fast as possible.
  - If the robot steps on a mine, the point loss is 100 and the game ends.
  - If the robot gets power ⚡️, it gains 1 point.
  - If the robot reaches the end goal, the robot gets 100 points.

- The problem is how do we train a robot to reach the end goal with the shortest path without stepping on a mine?

<p align="center">
  <img src="https://cdn-media-1.freecodecamp.org/images/3JXI06jyHegMS1Yx8rhIq64gkYwSTM7ZhD25" >
  <br>
  <i>Example problem</i>
</p>

#### **8.1.2 Q-Table**

- Basically, the Q-table is a look-up matrix where we have a row for each state and a column for each action. The value in the cell represents the reward for taking the action in the state.

- In Q-Table, the columns represent the actions, and the rows represent the states. Each Q-table score will be the maximum expected future reward that the robot will get if it takes that action at that state. This is an iterative process, as we need to improve the Q-Table at each iteration.

<p align="center">
  <img src="https://codelearn.io/Media/Default/Users/th1475369_40gmail_2Ecom/tictactoe_dqn/pic7.png" >
  <br>
  <i>Q-table</i>
</p>

- For the above example:

<p align="center">
  <img src="https://cdn-media-1.freecodecamp.org/images/AjVvggEquHgsnMN8i4N35AMfx53vZtELEL-l" >
  <br>
  <i>Q-table for above problem</i>
</p>

#### **8.1.3 Mathematics: the Q-Learning algorithm**

- The main process of Q-Learning algorithmn:

<p align="center">
  <img src="https://cdn-media-1.freecodecamp.org/images/oQPHTmuB6tz7CVy3L05K1NlBmS6L8MUkgOud" >
  <br>
  <i>Main process of Q-Learning algorithmn</i>
</p>

- Step 1: Initialize the Q-table with zeros.
  - There are n columns, where n is number of actions. There are m rows, where m is number of states.
  - We will initialise the values at 0.

<p align="center">
  <img src="https://cdn-media-1.freecodecamp.org/images/TQ9Wy3guJHUecTf0YA5AuQgB9yVIohgLXKIn" >
  <br>
  <i>Initialize the Q-table with zeros</i>
</p>

- Steps 2 and 3: choose and perform an action.
  - This combination of steps is done for an undefined amount of time. This means that this step runs until the time we stop the training, or the training loop stops as defined in the code.

  - In each step, we choose an action using the `epsilon-greedy policy`. This means that we either choose the action with the highest Q-value for the current state, or we choose a random action.

  - Epsilon-greedy policy based on epsilon value (in range 0 to 1). Then, we choose a random number between 0 and 1. If the random number is less than epsilon, we choose a random action. If the random number is greater than epsilon, we choose the action with the highest Q-value for the current state. So, if the epsilon value is 0.1, then 10% of the time, we will choose a random action, and 90% of the time, we will choose the action with the highest Q-value for the current state. It is a trade-off between exploration and exploitation. With higher epsilon values, we explore more, and with lower epsilon values, we exploit more.

  - In practice, we start with a higher epsilon value and then gradually decrease it as the training progresses. This is because we want to explore more in the beginning and exploit more towards the end of the training.

- Steps 4 and 5: evaluate.
  - Now we have taken an action and observed an outcome and reward.We need to update the function Q(s,a).

  <p align="center">
    <img src="https://cdn-media-1.freecodecamp.org/images/TnN7ys7VGKoDszzv3WDnr5H8txOj3KKQ0G8o" >
    <br>
    <i>Q-table update formula</i>
  </p>

  - Discount rate is a value between 0 and 1. It is used to balance immediate and future reward. Very low discount factor signifies importance to immediate reward while high discount signifies importance to future reward. The true value of the discount factor is application dependent but the optimal value of the discount factor lies between 0.2 to 0.8.

- We will repeat this again and again until the learning is stopped. In this way the Q-Table will be updated.

### **8.2 Deep Q Network (DQN)**

- As the name suggests, DQN is a Q-learning using Neural networks.

- The limitation of Q-learning is that if the state is too large, it will take a lot of space to store the Q-table and slow down the learning time because during the learning process the Q-table is accessed continuously.

- To solve such an issue, we can use a DQN algorithm. Where, instead of defining a Q-table, neural network approximates the Q-values for each action and state.

<p align="center">
  <img src="https://codelearn.io/Media/Default/Users/th1475369_40gmail_2Ecom/tictactoe_dqn/pic10.png" >
  <br>
  <i>Q-Learning vs Deep Q-Learning</i>
</p>

## **9. Reinforcement Learning vs Supervised Learning**

|Parameters|Reinforcement Learning|Supervised Learning|
|-|-|-|
|Decision style|reinforcement learning helps you to take your decisions sequentially.|In this method, a decision is made on the input given at the beginning.|
|Works on|Works on interacting with the environment.|Works on examples or given sample data.|
|Dependency on decision|In RL method learning decision is dependent. Therefore, you should give labels to all the dependent decisions.|Supervised learning the decisions which are independent of each other, so labels are given for every decision.|  
|Best suited|Supports and work better in AI, where human interaction is prevalent.|It is mostly operated with an interactive software system or applications.|
|Example|Chess game|Object recognition|

## **10. Why and When to use Reinforcement Learning?**

- Here are prime reasons for using Reinforcement Learning:
  - Helps to find which situation needs an action
  - Helps to discover which action yields the highest reward over the longer period.
  - Allows to figure out the best method for obtaining large rewards.

- Here are some conditions when you should not use reinforcement learning model.
  - When we have enough data to solve the problem with a supervised learning method
  - Reinforcement Learning is computing-heavy and time-consuming. in particular when the action space is large.
  - When the problem is simple and can be solved with a simple rule-based approach.
