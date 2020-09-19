import random
import time
import pandas as pd
import numpy as np
from buffer import ReplayBuffer
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import Adam

def dq_network(nb_actions,inputs_dim,hidden_size=32,hidden_layers=2): 
    i = Input(shape=(inputs_dim,))
    x=i
    for _ in range(hidden_layers): 
        x = Dense(hidden_size,activation='relu')(x)
    x=Dense(nb_actions)(x)
    model = Model(i,x)
    model.compile(loss='mse',optimizer='adam')
    print(model.summary())
    return model

class Q_Learning_agent(): 
    def __init__(self,name,size,lr,gamma,epsilon,batch_size,inputs_dim,fname,eps_dec=0.001,eps_min=0.01,memory_size=100000): 
        self.size=size
        self.name=name # Class name
        self.actions = ['right','left','up','down'] #Commun
        self.reward=0 #Commun
        self.discountFactor=gamma
        self.learning_rate=lr
        self.filename=fname #Commun 
        self.epsilon=epsilon
        self.batch_size=batch_size
        self.inputs_dim=inputs_dim
        self.memory_size=memory_size
        self.eps_dec=eps_dec #Commun
        self.eps_min=eps_min #Commun
        self.apple = ['N','S','E','W','NW','NE','SW','SE']
        self.sur=[[0,0,0,0],[1,1,1,1],[1,1,1,0],[1,1,0,1],[0,1,1,1],[1,0,1,1],[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        if self.name =='QL': 
            self.states=self.createStates()
            self.Q_table=self.createQTable()
            #self.Q_table=self.readModel()
        if self.name =='DQL': 
            self.memory = ReplayBuffer(self.memory_size,inputs_dim)
            self.q_eval = dq_network(len(self.actions),inputs_dim)
            #self.q_eval = self.readModel()


    def storeTransition(self,currentState,currentAction,reward,nextState,done):
        self.memory.store_transition(currentState,currentAction,reward,nextState,done)

    def updateEps(self): 
        if self.epsilon > self.eps_min: 
            self.epsilon-=self.eps_dec
        else: 
            self.epsilon=self.eps_min

    def readModel(self):
        if self.name == 'QL':
            data = pd.read_csv(str(self.filename))
            data.set_index('State',inplace=True)
            return data
        elif self.name =='DQL':
            model = load_model(str(self.filename))
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

    def getEps(self): 
        return self.epsilon

    def getQTable(self):
        return self.Q_table

    def SaveModel(self):
        if self.name == 'QL':
            self.getQTable().to_csv(str(self.filename),mode='w',header=True)
        elif self.name =='DQL':
            self.q_eval.save(str(self.filename)) 


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
            if self.name == 'QL': 
                r=np.argmax(self.getQTable().iloc[self.states.index(currentState)])
                action =self.actions[r]
            elif self.name == 'DQL': 
                a=self.q_eval.predict(np.asarray(currentState).reshape(1,-1))
                action = self.actions[np.argmax(a)]
        return action
    
    def train(self,currentState,action,nextState,reward,done):
        if self.name=='QL':
            if done ==False: 
                nextQValue =self.getNextQvalue(nextState)
            else: 
                nextQValue=0
            self.Q_table.iloc[self.states.index(currentState)][action]=(1-self.learning_rate)*(self.Q_table.iloc[self.states.index(currentState)][action])+self.learning_rate*(reward + (self.discountFactor)*(nextQValue))
            self.updateEps()
        elif self.name =='DQL': 
            self.storeTransition(np.asarray(currentState),action,reward,np.asarray(nextState),done)
            self.learn()
        