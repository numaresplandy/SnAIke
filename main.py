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
def saveData(score,bestGameRecord,path):
        df2 = pd.DataFrame(bestGameRecord,columns=['food','body'])
        df2.to_csv('bestGame/'+path+'.csv',mode='w',header=False)
        df = pd.DataFrame(score,columns=['game','score','mean_reward','eps','time','loss']) 
        df.to_csv('score/'+path+'.csv',mode='a',header=True,index=False)
    

def testScore(matrix,score):
    tmp = [row[1] for row in matrix]
    if all([score > x for x in tmp]):
        return True
    else: 
        return False

if __name__=='__main__': 
    tf.compat.v1.disable_eager_execution()
    size=200
    out=False
    run_max_time=5000
    n_games=1000
    batch = 512
    learning_rate = 0.0001
    layers =[[128,64]]
    for i in layers:
        tab_score=[]
        bestGameRecord=[]
        fnameModel= 'Qnet2_B512_Lr0.0001_H'+str(len(i))
        for j in i:
                fnameModel+='_'+str(j)
        #fnameModel='Qtable_lr_0.001'
        agent = deep_q_learning(learning_rate,size,batch_size=batch,inputs_dim=18,layer_size=i,fname=fnameModel)
        #agent = q_learning(learning_rate,size,fname=fnameModel)
        env = environment(size,run_max_time)
        #pygame.init()
        for game in tqdm(range(n_games)):
            done=False
            score=0
            sum_reward=0
            avg_loss=[0]
            currentState = env.reset(agent)
            #env.displayFunc(score,game,agent.getEps())
            while not done:
                action = agent.chooseAction(currentState)
                #env.displayFunc(score,game,agent.getEps())
                nextState, reward ,done = env.step(action,agent)
                loss = agent.train(currentState,action,nextState,reward,done,DoubleQ=False)
                currentState=nextState
                sum_reward+=reward
                if loss is not None:
                    avg_loss.append(loss)
                else:
                    avg_loss.append(0)
                #for event in pygame.event.get():
                #   if event.type == pygame.QUIT:
                #      done=True
                #     out=True
                if reward==env.reward[0]: 
                    score+=1
                if  game==0:
                    bestGameRecord = env.getFoodBody()
                elif testScore(tab_score,score)==True: 
                    bestGameRecord = env.getFoodBody()
            tab_score.append([game,score,np.sum(sum_reward),agent.getEps(),env.returnTime(),np.mean(avg_loss)])
            if out==True:
                break
        if out==False:
            saveData(tab_score,bestGameRecord,fnameModel)
            agent.saveModel()

            

            

    


