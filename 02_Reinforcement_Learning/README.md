# **Reinforcement Learning**

## **1. What is Reinforcement Learning?**

- Reinforcement learning is a machine learning training method based on rewarding desired behaviors and punishing undesired ones.
- In general, a reinforcement learning agent -- the entity being trained -- is able to perceive and interpret its environment, take actions and learn through trial and error.

- Key Features of Reinforcement Learning:
  - The agent is not instructed about the environment and what actions need to be taken.
  - RL based on the hit and trial process.
  - The agent takes the next action and changes states according to the feedback of the previous action.
  - Feedback is always delayed, not instantaneous
  - The environment is stochastic, and the agent needs to explore it to reach to get the maximum positive rewards.

## **2. Terms used in Reinforcement Learning**

- **Agent**: An entity that can perceive/explore the environment and act upon it.

- **Environment (e)**: A scenario that an agent is present or surrounded by. In RL, we assume the stochastic environment, which means it is random in nature.

- **Action**: the set of all possible moves that an agent can take in a given situation within the environment.

- **State (s)** is a situation returned by the environment after each action taken by the agent.

- **Reward (R)**: A feedback returned to the agent from the environment to evaluate the action of the agent.

- **Policy (π)**: It is a strategy which applies by the agent to decide the next action based on the current state.

- **Value (V)**:  It is expected long-term retuned with the discount factor and opposite to the short-term reward.

- **Q value or action value (Q)**: It is mostly similar to the value, but it takes one additional parameter as a current action (a).
  
![RL](https://www.guru99.com/images/1/082319_0514_Reinforceme1.png)

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

- A policy can be defined as a way how an agent behaves at a given time. It maps the perceived states of the environment to the actions taken on those states. A policy is the core element of the RL as it alone can define the behavior of the agent. In some cases, it may be a simple function or a lookup table, whereas, for other cases, it may involve general computation as a search process. It could be deterministic or a stochastic policy:

- For deterministic policy:

  ```math
  a = π(s)
  ```

- For stochastic policy:

  ```math
  π(a|s) = P[A=a|S=s]
  ```

### **4.2 Reward function**

- The goal of reinforcement learning is defined by the reward signal. At each state, the environment sends an immediate signal to the learning agent, and this signal is known as a reward signal. These rewards are given according to the good and bad actions taken by the agent.

- The agent's main objective is to maximize the total number of rewards for good actions. The reward signal can change the policy, such as if an action selected by the agent leads to low reward, then the policy may change to select other actions in the future.

### **4.3 Value function**

- The value function gives information about how good the situation and action are and how much reward an agent can expect. A reward indicates the immediate signal for each good and bad action, whereas a value function specifies the good state and action for the future.

- The value function depends on the reward as, without reward, there could be no value. The goal of estimating values is to achieve more rewards.

### **4.4 Model**

- The models mimics the behavior of the environment. With the help of the model, one can make inferences about how the environment will behave. Such as, if a state and an action are given, then a model can predict the next state and reward.

- The model is used for planning, which means it provides a way to take a course of action by considering all future situations before actually experiencing those situations. The approaches for solving the RL problems with the help of the model are termed as the model-based approach. Comparatively, an approach without using a model is called a model-free approach.

## **5. How does Reinforcement Learning work?**

### **5.1 Example**

- Consider the following problems:

![prob](https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-works.png)

- In the above problem, the agent is at s9 and it have to get to s4 for the diamond.
  - The agent can move in four directions: up, down, left, and right.
  - If it reach s4 (diamond), then it will get a reward of +1.
  - If it reach s8 (fire), then it will get a reward of -1.
  - The agent can take any path to reach the diamond, but it needs to make it in minimum steps.

- Initally, the agent will explore the enviroment and try to reach the diamond. As soon as it reachs the diamond, it will backtrace its step back and mark values of all states which leads towards the goal as V = 1.

![Without Bellman](https://media.geeksforgeeks.org/wp-content/uploads/20210914203249/Env11.png)

- But, if we changr the start position, the agent can not find the path to the goal. So, we need to use the Bellman equation to solve this problem.

![Problem](https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-works3.png)

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

  ![Results](https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-bellman-equation.png)

  - Now we move to s7 block, here the agent have 3 option:
    - UP to s3
    - LEFT to s8 (fire)
    - DOWN to s11
    - Can not move left to s6 because it is a wall.

    ![s7](https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-bellman-equation2.png)

  - The `max` in Bellman equation denote the most optimal path among all posible action that agent can take at a given state. Among all these actions available the maximum value for that state is the UP action. So, the Bellman equation will be:

    ```math
    V(s7) = max[R(s7) + γV(s')] = max[0 + 0.9 * 1] = 0.9
    ```

  - Continue the process, we got the final result. The agent will take the path with maximum value by following the increasing value of the states based on the Bellman equation.

  ![Results](https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-bellman-equation3.png)

## **6. Types of Reinforcement Learning**

There are two types of reinforcement learning methods.

### **6.1 Positive**

- The positive reinforcement learning means adding something to increase the tendency that expected behavior would occur again. It impacts positively on the behavior of the agent and increases the strength of the behavior.

- This type of Reinforcement helps you to maximize performance and sustain change for a more extended period. However, too much Reinforcement may lead to over-optimization of state, which can affect the results.

### **6.2 Negative**

- The negative reinforcement learning is opposite to the positive reinforcement as it increases the tendency that the specific behavior will occur again by avoiding the negative condition.

- It can be more effective than the positive reinforcement depending on situation and behavior, but it provides reinforcement only to meet minimum behavior.

## **6. Learning Models of Reinforcement**

- There are two important learning models in reinforcement learning:
  - Markov Decision Process
  - Q learning

### **6.1 Markov Decision Process**

- The following parameters are used to get a solution:
  - Set of actions - A
  - Set of states - S
  - Reward - R
  - Policy - n
  - Value - V

- The mathematical approach for mapping a solution in reinforcement Learning is recon as a Markov Decision Process or (MDP).

![Markov](https://www.guru99.com/images/1/082319_0514_Reinforceme3.png)

### **6.2 Q learning**

- Q learning is a value-based method of supplying information to inform which action an agent should take.

- Let’s understand this method by the following example:
  - There are five rooms in a building which are connected by doors.
  - Each room is numbered 0 to 4
  - The outside of the building can be one big outside area (5)
  - Doors number 1 and 4 lead into the building from room 5

![Q](https://www.guru99.com/images/1/082319_0514_Reinforceme4.png)

- Next, you need to associate a reward value to each door:
  - Doors which lead directly to the goal have a reward of 100
  - Doors which is not directly connected to the target room gives zero reward
  - As doors are two-way, and two arrows are assigned for each room
  - Every arrow in the above image contains an instant reward value

- Explanation:
  - In this image, you can view that room represents a state
  - Agent’s movement from one room to another represents an action
  - In the below-given image, a state is described as a node, while the arrows show the action.

![Q](https://www.guru99.com/images/1/082319_0514_Reinforceme5.png)

- For example, an agent traverse from room number 2 to 5
  - Initial state = state 2
  - State 2-> state 3
  - State 3 -> state (2,1,4)
  - State 4-> state (0,5,3)
  - State 1-> state (5,3)
  - State 0-> state 4



## **8. Reinforcement Learning vs Supervised Learning**

|Parameters|Reinforcement Learning|Supervised Learning|
|-|-|-|
|Decision style|reinforcement learning helps you to take your decisions sequentially.|In this method, a decision is made on the input given at the beginning.|
|Works on|Works on interacting with the environment.|Works on examples or given sample data.|
|Dependency on decision|In RL method learning decision is dependent. Therefore, you should give labels to all the dependent decisions.|Supervised learning the decisions which are independent of each other, so labels are given for every decision.|  
|Best suited|Supports and work better in AI, where human interaction is prevalent.|It is mostly operated with an interactive software system or applications.|
|Example|Chess game|Object recognition|

## **9. Why use and When not to use Reinforcement Learning?**

- Here are prime reasons for using Reinforcement Learning:
  - It helps you to find which situation needs an action
  - Helps you to discover which action yields the highest reward over the longer period.
  - Reinforcement Learning also provides the learning agent with a reward function.
  - It also allows it to figure out the best method for obtaining large rewards.

- You can’t apply reinforcement learning model is all the situation. Here are some conditions when you should not use reinforcement learning model.
  - When you have enough data to solve the problem with a supervised learning method
  - You need to remember that Reinforcement Learning is computing-heavy and time-consuming. in particular when the action space is large.

- Here are the major challenges you will face while doing Reinforcement earning:
  - Feature/reward design which should be very involved
  - Parameters may affect the speed of learning.
  - Realistic environments can have partial observability.
  - Too much Reinforcement may lead to an overload of states which can diminish the results.
  - Realistic environments can be non-stationary.

## **10. Implementing Reinforcement Learning**
