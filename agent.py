import random
import time
import pandas as pd
import numpy as np
from buffer import ReplayBuffer
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import Adam

def dq_network(nb_actions,inputs_dim,hidden_size=64,hidden_layers=2): 
    i = Input(shape=(inputs_dim,))
    x=i 
    x = Dense(32,activation='relu')(x)
    x = Dense(16,activation='relu')(x)
    x=Dense(nb_actions)(x)
    model = Model(i,x)
    model.compile(loss='mse',optimizer='adam')
    print(model.summary())
    return model

class agent(): 
    def __init__(self,size,fname): 
        self.actions = ['right','left','up','down'] #Commun
        self.apple = ['N','S','E','W','NW','NE','SW','SE']
        self.reward=0 #Commun
        self.filename=fname #Commun
        self.size=size
        self.epsilon=0
        self.eps_dec=0.0001 #Commun
        self.eps_min=0.1 #Commun
        self.learning_rate=0.1
        self.discountFactor=0.90

    def updateEps(self): 
        if self.epsilon > self.eps_min: 
            self.epsilon-=self.eps_dec
        else: 
            self.epsilon=self.eps_min
    
    def getEps(self): 
        return self.epsilon

    def getApplePosition(self,posHead,foodspw):
        apple = ''
        posFood = foodspw.getFoodPos()
        if posFood[0]==posHead[0] and posFood[1] in range(0,posHead[1]):
            apple='N'
        elif posFood[0]==posHead[0] and posFood[1] in range(posHead[1],(self.size)+10): 
            apple='S' 
        elif posFood[1]==posHead[1] and posFood[0] in range(0,posHead[0]): 
            apple='W'
        elif posFood[1]==posHead[1] and posFood[0] in range(posHead[0],(self.size)+10): 
            apple='E'
        elif posFood[1] in range(0,posHead[1]) and posFood[0] in range(0,posHead[0]): 
            apple='NW' 
        elif  posFood[1] in range(0,posHead[1]) and posFood[0] in range(posHead[0],(self.size)+10): 
            apple='NE'
        elif  posFood[1] in range(posHead[1],(self.size)+10) and posFood[0] in range(0,posHead[0]): 
            apple='SW'
        elif  posFood[1] in range(posHead[1],(self.size)+10) and posFood[0] in range(posHead[0],(self.size)+10): 
            apple='SE'
        return self.apple.index(apple) 

class deep_q_learning(agent):
    def __init__(self,size,batch_size,inputs_dim,fname,memory_size=100000):
         agent.__init__(self,size,fname)
         self.batch_size=batch_size
         self.inputs_dim=inputs_dim
         self.memory_size=memory_size
         self.memory = ReplayBuffer(self.memory_size,inputs_dim)
         #self.q_eval = dq_network(len(self.actions),inputs_dim)
         self.q_eval = self.readModel()

    def storeTransition(self,currentState,currentAction,reward,nextState,done):
        self.memory.store_transition(currentState,currentAction,reward,nextState,done)
    
    def readModel(self):
        model = load_model('model/'+str(self.filename)+'.h5')
        return model

    def learn(self): 
        if self.memory.mem_cntr <self.batch_size: 
            return 
        s,a,r,s_,d= self.memory.sample_buffer(self.batch_size)
        q_eval = self.q_eval.predict(s)
        q_next = self.q_eval.predict(s_)
        q_target = np.copy(q_eval)
        batch_i = np.arange(self.batch_size)
        q_target[batch_i,a]=r+self.discountFactor * np.max(q_next,axis=1)*d
        self.q_eval.train_on_batch(s,q_target) 
        self.updateEps()
    
    def saveModel(self):
        self.q_eval.save('model/'+str(self.filename)+'.h5') 
    
    def chooseAction(self,currentState): 
        if np.random.random() < self.epsilon: 
            action = np.random.choice(self.actions)
        else:
            a=self.q_eval.predict(np.asarray(currentState).reshape(1,-1))
            action = self.actions[np.argmax(a)]
        return action
    
    def train(self,currentState,action,nextState,reward,done):
        self.storeTransition(np.asarray(currentState),action,reward,np.asarray(nextState),done)
        self.learn()
    
    def getState(self,head,body,foodspw):
        surrounwding=np.zeros(24,dtype=np.int32)
        index=0
        for i in range(-20, 30, 10):
            for j in range(-20, 30, 10):
                if [head[0]+j,head[1]+i] in body and ([head[0]+j,head[1]+i] != head): 
                    surrounwding[index]=1
                if head[1]+i <10:
                    surrounwding[index]=1
                if head[1]+i >self.size:
                    surrounwding[index]=1
                if head[0]+j <10:
                    surrounwding[index]=1 
                if head[0]+j > self.size:
                    surrounwding[index]=1 
                if ([head[0]+j,head[1]+i] != head): 
                    index+=1
        return np.hstack([surrounwding,self.getApplePosition(head,foodspw)]).tolist()
        

class q_learning(agent):
    def __init__(self,size,fname):
         agent.__init__(self,size,fname)
         self.sur=[[0,0,0,0],[1,1,1,1],[1,1,1,0],[1,1,0,1],[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
         self.states=self.createStates()
         #self.Q_table=self.createQTable()
         self.Q_table=self.readModel()

    def readModel(self):
        data = pd.read_csv('qtable/'+str(self.filename)+'.csv')
        data.set_index('State',inplace=True)
        return data
    
    def getQTable(self):
        return self.Q_table

    def saveModel(self):
        self.getQTable().to_csv('qtable/'+str(self.filename)+'.csv',mode='w',header=True)
    
    def createStates(self): 
        states=[]
        for i in self.sur:
            for t in self.apple:
                states.append(np.hstack([i,self.apple.index(t)]).tolist())
        return states
        
    def createQTable(self): 
        table = np.zeros((len(self.states),5))
        for i in range(len(self.states)): 
            table[i][0]=i
        df = pd.DataFrame(table,columns=['State','right','left','up','down']) 
        df.set_index('State',inplace=True)
        return df
        
    def getNextQvalue(self,nextState):
        tmp=0
        for i in self.actions: 
            tmp+=self.getQTable().iloc[self.states.index(nextState)][i]
        if tmp==0: 
            r =random.randint(0,3)
            return self.getQTable().iloc[self.states.index(nextState)][r]
        else:
            r=np.argmax(self.getQTable().iloc[self.states.index(nextState)])
            return self.getQTable().iloc[self.states.index(nextState)][r]

    def chooseAction(self,currentState): 
        if np.random.random() < self.epsilon: 
            action = np.random.choice(self.actions)
        else: 
            r=np.argmax(self.getQTable().iloc[self.states.index(currentState)])
            action =self.actions[r]
        return action
    
    def train(self,currentState,action,nextState,reward,done):
        if done ==False: 
            nextQValue =self.getNextQvalue(nextState)
        else: 
            nextQValue=0
        self.Q_table.iloc[self.states.index(currentState)][action]=(1-self.learning_rate)*(self.Q_table.iloc[self.states.index(currentState)][action])+self.learning_rate*(reward + (self.discountFactor)*(nextQValue))
        self.updateEps()

    def getState(self,pos,body,foodspw):
        surrounwding=[0,0,0,0]
        if [pos[0],pos[1]-10] in body or pos[1]==10:
            #DOWN
            surrounwding[0]=1
        if [pos[0],pos[1]+10] in body or pos[1]==self.size: 
            #UP
            surrounwding[1]=1
        if [pos[0]+10,pos[1]] in body or pos[0]==self.size: 
            #LEFT
            surrounwding[2]=1
        if [pos[0]-10,pos[1]] in body or pos[0]==10: 
            #RIGHT
            surrounwding[3]=1
        #return [surrounwding,self.getApplePosition(pos,foodspw)]
        return np.hstack([surrounwding,self.getApplePosition(pos,foodspw)]).tolist()

