import random
import time
import pandas as pd
import numpy as np
#import pygame
from environment import environment
from agent import agent
from agent import deep_q_learning
from agent import q_learning
from tqdm import tqdm
import tensorflow as tf

#saving Score and Epsilon value
def saveScore(score,path):
    df = pd.DataFrame(score,columns=['game','score','mean_reward','eps','time']) 
    df.to_csv('score/'+path+'.csv',mode='a',header=False)


if __name__=='__main__': 
    tf.compat.v1.disable_eager_execution()
    size=200 #Window size
    n_games=200#Number of game to play
    run_max_time=1000 #To avoid infinite loops
    score=0
    rewardID =1
    out=False
    tab_score=[]
    sum_reward=[]
    fnameModel='Qnet7_train_e_5000_lr_0.1_df_0.9'
    fnameTest='Qnet7_test_e_200_lr_0.1_df_0.9'
    agent = deep_q_learning(size,batch_size=64,inputs_dim=25,fname=fnameModel)
    #agent = q_learning(size,fname=fnameModel)
    env = environment(size,run_max_time,rewardID)
    #pygame.init()

    for game in tqdm(range(n_games)):
        done=False
        score=0
        sum_reward=[]
        currentState = env.reset(agent)
        #env.displayFunc(score,game,agent.getEps())
        while not done:
            action = agent.chooseAction(currentState)
            #env.displayFunc(score,game,agent.getEps())
            nextState, reward ,done = env.step(action,agent)
            agent.train(currentState,action,nextState,reward,done)
            currentState=nextState
            sum_reward.append(reward)
            #for event in pygame.event.get():
            #if event.type == pygame.QUIT:
            #   done=True
            #  out=True
            if reward==env.reward[0]: 
                score+=1
        tab_score.append([game,score,np.sum(sum_reward),agent.getEps(),env.returnTime()])
        if out==True:
            break
    if out==False:
        pass
        agent.saveModel()
        saveScore(tab_score,fnameTest)
        


            


        

            

            

    


