# SnAIke

I remenber me, 10 years old playing again and again to snake on my Nokia 3310, trying to reach a better score each time. 14 years later and a engineering degree i accept the challenge again. In this project i have used some reinforcement learning alogrithm to autonmate the game snake. Follow me on this journey! 

## Q-learning Algorithm

To start this adventure i have implemented a basic Q-learning algortihm. This algortithm is based on the idea that we are trying give a value to a each possible pair of states-actions of an agent on the envirnonement. Let's break the terminologies : 
- The environement : In reinforcement learning the environment is the space in wich the action take place. In our case it's the grid (20x20) of the game.
- The agent : The agent represent the algorithm, it will try to learn how to take actions on the environment to acheive it's goal. Here the agent is the snake itself.
- A state : It's the information that the agent get from the environment. In a way the states are the "vision" of the agent. At each time the agent is in a certain state in the environement. For our Q-learning algortithm we have define the states as follow : 

| State | Description |
| ------ | ----------- |
| Surownding Vision   | path to data files to supply the data that will be passed into templates. |
| Direction of the apple | North, North-west, North-east, South, South-west, South-east, West, East |

- The actions : In the game the snake can take four possible actions : Up, Down, Left, right
- The rewards :  This will help the agent learn overtime. The rewards map the relevance of the action taken by the agent to go from state s to state s'. In our case i've decided to define the rewards like: 
    - The snake eat the apple : +10 
    - The snake die : -100 
    - The snake is getting closer to the apple : +1 
    - The snake is getting far away of the apple : -1 

So the goal here is to find a function that will give a value for each possible State-Action pair. This value is called the Q-value. We define the Q-function as follow : 

\alpha


The particularity of the function which came from the Bellan's equation, is that at each time step (each time the snake is in a state and take an action) we will update the Q-value in function of 
