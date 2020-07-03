import random
import time
import pandas as pd
import numpy as np
#import pygame
from environment import environment
from q_learning_agent import Q_Learning_agent
from tqdm import tqdm
import tensorflow as tf

#saving Score and Epsilon value
def SaveScore(score,path):
    df = pd.DataFrame(score,columns=['game','score','mean_reward','eps','time']) 
    df.to_csv(path,mode='w',header=True)


if __name__=='__main__': 
    tf.compat.v1.disable_eager_execution()
    size=400 #Window size
    n_games=500#Number of game to play
    run_max_time=3000 #To avoid infinite loops
    score=0
    out=False
    tab_score=[]
    mean_reward=[]
    #Choose 'QL' for classic Q_learning or 'DQL' for Deep Q netword
    name='DQL'
    #fnameModel='/Users/numa/Desktop/Numa/Projets/Local/Snake/model/Q_net_2_gamma_0.98.h5'
    fnameModel='model/Qnet2_H1_256.h5'
    fnameScore='score/Qnet2_H1_256_train_500_epoch.csv'
    agent = Q_Learning_agent(name,size,lr=0.1,gamma=0.98,epsilon=1.0,batch_size=32,inputs_dim=25,eps_dec=0.001,eps_min=0.01,memory_size=100000,fname=fnameModel)
    env = environment(size,run_max_time,name)
   # pygame.init()

    for game in tqdm(range(n_games)): 
        done=False
        score=0
        mean_reward=[]
        currentState = env.reset()
        #env.displayFunc(score,game,agent.getEps())
        while not done:
            action = agent.chooseAction(currentState)
            #env.displayFunc(score,game,agent.getEps())
            nextState, reward ,done = env.step(action)
            agent.train(currentState,action,nextState,reward,done)
            currentState=nextState
            mean_reward.append(reward)
            #for event in pygame.event.get():
              #if event.type == pygame.QUIT:
                 #done=True
                 #out=True
            if reward==50: 
                score+=1
        tab_score.append([game,score,np.mean(mean_reward),agent.getEps(),env.returnTime()])
        if out==True: 
            break
    if out==False:
        agent.SaveModel()
        SaveScore(tab_score,fnameScore)


            


        

            

            

    


