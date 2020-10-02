import random
import time
import pandas as pd
import numpy as np
import pygame
from environment import environment
from agent import agent
from agent import deep_q_learning
from agent import q_learning
from tqdm import tqdm
import tensorflow as tf

#saving Score and Epsilon value
def saveData(score,bestGameRecord,pathTrain,pathTest,test):
    if test==False:
        df = pd.DataFrame(score,columns=['game','score','mean_reward','eps','time']) 
        df.to_csv('score/'+pathTrain+'.csv',mode='a',header=False)
    if test==True:
        df2 = pd.DataFrame(bestGameRecord,columns=['food','body'])
        df2.to_csv('bestGame/'+pathTest+'.csv',mode='w',header=False)
        df = pd.DataFrame(score,columns=['game','score','mean_reward','eps','time']) 
        df.to_csv('score/'+pathTest+'.csv',mode='a',header=False)
    

def testScore(matrix,score):
    tmp = [row[1] for row in matrix]
    if all([score > x for x in tmp]):
        return True
    else: 
        return False

if __name__=='__main__': 
    tf.compat.v1.disable_eager_execution()
    test=True
    size=200 #Window size
    rewardID =1
    out=False
    tab_score=[]
    sum_reward=[]
    bestGameRecord=[]
    score=0
    fnameModel='Qtable_train_e_5000_lr_0.1_df_0.9'
    fnameTest='Qnet2_test_e_200_lr_0.1_df_0.99'
    if test == True: 
        #set up for test mode
        run_max_time=10000
        n_games=200
    else: 
        #Set up for train mode
        run_max_time=1200
        n_games=2000
    #agent = deep_q_learning(test,size,batch_size=64,inputs_dim=26,fname=fnameModel)
    agent = q_learning(test,size,fname=fnameModel)
    env = environment(size,run_max_time,rewardID)
    pygame.init()


    for game in tqdm(range(n_games)):
        done=False
        score=0
        sum_reward=[]
        currentState = env.reset(agent)
        env.displayFunc(score,game,agent.getEps())
        while not done:
            action = agent.chooseAction(currentState)
            env.displayFunc(score,game,agent.getEps())
            nextState, reward ,done = env.step(action,agent)
            if test ==False:
                agent.train(currentState,action,nextState,reward,done)
            currentState=nextState
            sum_reward.append(reward)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done=True
                    out=True
            if reward==env.reward[0]: 
                score+=1
        if test==True:
            if  game==0:
                bestGameRecord = env.getFoodBody()
            elif testScore(tab_score,score)==True: 
                bestGameRecord = env.getFoodBody()

        tab_score.append([game,score,np.sum(sum_reward),agent.getEps(),env.returnTime()])
        if out==True:
            break
    if out==False:
        saveData(tab_score,bestGameRecord,fnameModel,fnameTest,test)
        agent.saveModel()
        
        
        
        


            


        

            

            

    


