# SnAIke

I remenber me, 10 years old playing again and again to snake on my Nokia 3310, trying to reach a better score each time. 14 years later and a engineering degree i accept the challenge again. In this project i have used some reinforcement learning alogrithm to autonmate the game snake. Follow me on this journey! 

## Q-learning Algorithm

### Environment, Agent, States, rewards ... What's this all about ?

To start this adventure i have implemented a basic Q-learning algortihm. This algortithm is based on the idea that we are trying give a value to a each possible pair of states-actions of an agent on the envirnonement. Let's break the terminologies : 
- **The environement** : In reinforcement learning the environment is the space in wich the action take place. In our case it's the grid (20x20) of the game.
- **The agent** : The agent represent the algorithm, it will try to learn how to take actions on the environment to acheive it's goal. Here the agent is the snake itself.
- **A state** : It's the information that the agent get from the environment. In a way the states are the "vision" of the agent. At each time the agent is in a certain state in the environement. For our Q-learning algortithm we have define the states as follow : 

| State | Type | Description |
| ------ | --- |----------- |
| Surownding Vision   | Arrat of integer of length 4| 0: There is Nothing <br/> 1: There is a wall or a part of the tail  |
| Direction of the apple | integer between 0 and 7  | 0: North <br/> 1: North-west <br/> 2: North-east <br/> 3: South <br/> 4: South-west <br/> 5: South-east <br/> 6: West <br/> 7: East |

Each state is composed of the surwonfing vision and the direction of the apple relative to the head snake, so we have : 2^4 * 8 = 128 possibles states.

- **The actions** : In the game the snake can take four possible actions : Up, Down, Left, right
- **The rewards** :  This will help the agent learn overtime. The rewards map the relevance of the action taken by the agent to go from state s to state s'. In our case i've decided to define the rewards like: 

| Reward Description | Value |
| ------ | ----------- |
| The snake eat the apple | +10 |
| The snake die | -100 |
| The snake is getting closer to the apple | +1 |
| The snake is getting far away of the apple | -1 |

### The algorithm 

So now with all this information let's see how it is going on the math side. The goal of the algorithm is to fill a table call a the "state-action table" with some value that will give us the relevance of the actions depending on the states. The table will look like that: 

| States | Up | Down | Right | Left |
| ------ | ---- | ---- | ---- | ---- |
| s0 | 12.5 | 1.5 | 2.9 | 24.5 |
| s1 | 6.8 | 12.5 | 5.9 | 0.5 |
| ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ...  |
| s127 | 15.5 | 1.4 | 32 | 9 |


To fill the table we are using the function called the Q-function. This  function which came from the Bellan's equation, is that at each time step (each time the snake is in a state and take an action) we will update the Q-value in function of 
