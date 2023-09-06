# **Reinforcement Learning**

## **1. What is Reinforcement Learning?**

- Reinforcement Learning is a type of Machine Learning, and thereby also a branch of Artificial Intelligence. It allows machines and software agents to automatically determine the ideal behavior within a specific context, in order to maximize its performance. Simple reward feedback is required for the agent to learn its behavior; this is known as the reinforcement signal.

## **2. Components of Reinforcement Learning**

Some important components in RL:

- Agent: It is an assumed entity which performs actions in an environment to gain some reward.
- Environment (e): A scenario that an agent has to face.
- Reward (R): An immediate return given to an agent when he or she performs specific action or task.
- State (s): State refers to the current situation returned by the environment.
- Policy (π): It is a strategy which applies by the agent to decide the next action based on the current state.
- Value (V): It is expected long-term return with discount, as compared to the short-term reward.
- Value Function: It specifies the value of a state that is the total amount of reward. It is an agent which should be expected beginning from that state.
- Model of the environment: This mimics the behavior of the environment. It helps you to make inferences to be made and also determine how the environment will behave.
- Model based methods: It is a method for solving reinforcement learning problems which use model-based methods.
- Q value or action value (Q): Q value is quite similar to value. The only difference between the two is that it takes an additional parameter as a current action.
  
![RL](https://www.guru99.com/images/1/082319_0514_Reinforceme1.png)

## **3. How does Reinforcement Learning work?**

- Consider the scenario of teaching new tricks to your cat:
  - As cat doesn’t understand English or any other human language, we can’t tell her directly what to do. Instead, we follow a different strategy.
  - We emulate a situation, and the cat tries to respond in many different ways. If the cat’s response is the desired way, we will give her fish.
  - Now whenever the cat is exposed to the same situation, the cat executes a similar action with even more enthusiastically in expectation of getting more reward(food).
  - That’s like learning that cat gets from “what to do” from positive experiences.
  - At the same time, the cat also learns what not do when faced with negative experiences.

- Example of Reinforcement Learning

![RL](https://www.guru99.com/images/1/082319_0514_Reinforceme2.png)

- In this case:
  - Your cat is an agent that is exposed to the environment. In this case, it is your house. An example of a state could be your cat sitting, and you use a specific word in for cat to walk.
  - Our agent reacts by performing an action transition from one “state” to another “state.”
  - For example, your cat goes from sitting to walking.
  - The reaction of an agent is an action, and the policy is a method of selecting an action given a state in expectation of better outcomes.
  - After the transition, they may get a reward or penalty in return.

## **4. Reinforcement Learning Algorithms**

There are three approaches to implement a Reinforcement Learning algorithm.

### **4.1 Value-Based**

- In a value-based Reinforcement Learning method, you should try to maximize a value function V(s). In this method, the agent is expecting a long-term return of the current states under policy π.

### **4.2 Policy-based**

- In a policy-based RL method, you try to come up with such a policy that the action performed in every state helps you to gain maximum reward in the future.
- Two types of policy-based methods are:
  - Deterministic: For any state, the same action is produced by the policy π.
  - Stochastic: Every action has a certain probability, which is determined by the following equation. Stochastic Policy:

  ```math
  π(a|s) = P[A=a|S=s]
  ```

### **4.3 Model-Based**

- In this Reinforcement Learning method, you need to create a virtual model for each environment. The agent learns to perform in that specific environment.

## **5. Types of Reinforcement Learning**

There are two types of reinforcement learning methods.

### **5.1 Positive**

- It is defined as an event, that occurs because of specific behavior. It increases the strength and the frequency of the behavior and impacts positively on the action taken by the agent.

- This type of Reinforcement helps you to maximize performance and sustain change for a more extended period. However, too much Reinforcement may lead to over-optimization of state, which can affect the results.

### **5.2 Negative**

- Negative Reinforcement is defined as strengthening of behavior that occurs because of a negative condition which should have stopped or avoided. It helps you to define the minimum stand of performance. However, the drawback of this method is that it provides enough to meet up the minimum behavior.

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

## **7. Characteristics of Reinforcement Learning**

- Here are important characteristics of reinforcement learning
  - There is no supervisor, only a real number or reward signal
  - Sequential decision making
  - Time plays a crucial role in Reinforcement problems
  - Feedback is always delayed, not instantaneous
  - Agent’s actions determine the subsequent data it receives

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
