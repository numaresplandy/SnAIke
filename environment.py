import random
import time
import pandas as pd
import numpy as np
#import pygame

#Snake / FoodSpawner / Environment 

class Snake(): 
    def __init__(self,size): 
        self.size=size
        self.position = self.getFoodCoordonate()
        self.body = [self.position]
        self.direction = self.changeDirTo(0)
        self.changeDirectionTo = self.direction


    def getFoodCoordonate(self): 
        spnX=((random.randrange(1,self.size/10)*10)+10)
        spnY=((random.randrange(1,self.size/10)*10)+10)
        return [spnX,spnY]


    def changeDirTo(self,dir):
        direc=['right','left','up','down']
        if dir=='right' and not self.direction=="left": 
            self.direction="right"
        if dir=='left' and not self.direction=="right":
            self.direction='left'
        if dir=="up" and not self.direction=='down':
            self.direction='up'
        if dir=='down' and not self.direction=='up':
            self.direction='down'
        if dir==0: 
            return direc[random.randint(0,3)]


    def move(self,food_position):
        if self.direction=='right': 
            self.position[0]+=10
        if self.direction=='left': 
            self.position[0]-=10
        if self.direction=='up': 
            self.position[1]-=10
        if self.direction=='down': 
            self.position[1]+=10
        self.body.insert(0,list(self.position))
        if self.position==food_position:
            return 1
        else:
            self.body.pop()
            return 0


    def checkColission(self):
        if self.position[0] >self.size or self.position[0]<10:
            return 1
        elif self.position[1] >self.size or self.position[1] <10:
            return 1
        for body in self.body[1:]:
            if self.position == body:
                return 1
        return 0

    def getHeadPos(self):
        return self.position
    
    def getBody(self):
        return self.body


class FoodSpawner():
    def __init__(self,size,body):
        self.size=size
        self.position=self.getFoodCoordonate(body)
        self.isFoodOnScreen = True

    def SpawnFood(self,body):
        if self.isFoodOnScreen == False:
            self.position = self.getFoodCoordonate(body)
            self.isFoodOnScreen = True
        return self.position
    def getFoodCoordonate(self,body): 
        Tries=True
        while Tries: 
            spnX=((random.randrange(1,self.size/10)*10)+10)
            spnY=((random.randrange(1,self.size/10)*10)+10)
            if [spnX,spnY] not in body:
                Tries=False
        return [spnX,spnY]

    def setFoodOnScreenTo(self,b):
        self.isFoodOnScreen = b

    def getFoodPos(self):
        return self.position


class environment(): 
    def __init__(self,size,run_max_time,rewardID):
        self.reward = self.rewardFunc(rewardID) 
        self.size=size
        self.run_max_time = run_max_time
        self.snake=Snake(size)
        self.foodSpawner=FoodSpawner(size,self.snake.getBody())
        self.foodPosition=self.foodSpawner.SpawnFood(self.snake.getBody())
        self.currentState=[]
        self.nextState=[]
        self.distance=0
        self.apple = ['N','S','E','W','NW','NE','SW','SE']
        self.time=0
        #self.fps = pygame.time.Clock()
        #self.win = pygame.display.set_mode((self.size+20,self.size+20))


    def reset(self,agent):
        self.snake=Snake(self.size)
        self.foodSpawner=FoodSpawner(self.size,self.snake.getBody())
        self.foodPosition=self.foodSpawner.SpawnFood(self.snake.getBody())
        self.nextState=[]
        self.distance=0
        self.time=0
        self.currentState=agent.getState(self.snake.getHeadPos(),self.snake.getBody(),self.foodSpawner,self.distance)
        return self.currentState


    def step(self,action,agent):
        self.snake.changeDirTo(action)
        self.foodPosition = self.foodSpawner.SpawnFood(self.snake.getBody())
        self.time+=1
        done=False
        if(self.snake.move(self.foodPosition)==1): 
            self.foodSpawner.setFoodOnScreenTo(False)
            reward = self.giveReward(0)
        elif(self.snake.checkColission()==1) or self.time==self.run_max_time:
            self.foodSpawner.setFoodOnScreenTo(False) 
            reward =self.giveReward(1)
            done=True
        else:
            reward = self.giveReward(3)
        self.next_state=agent.getState(self.snake.getHeadPos(),self.snake.getBody(),self.foodSpawner,self.distance)
        return self.next_state, reward, done

    def updateStates(self): 
        self.currentState=self.nextState
        self.nextState=[]

    def returnTime(self):
        return self.time

    def rewardFunc(self,rID): 
        switcher = {
                1: [40,-100,-5,5],
                2: [100,-100,0,0],
                3: [100,-100,-1,1],
                4: [100,-50,-10,10],
                5: [100,-100,-5,5],
                6: [30,-100,-1,1]
            } 
        return switcher.get(rID)

    def giveReward(self,ID):
        a = self.distanceHeadApple()
        if ID==0: #touch the apple
            return self.reward[0]
        if ID==1:  #touch a wall or itself
            return self.reward[1]
        if ID==3 and a ==2: #getting far away of the apple
            return self.reward[2]
        if ID==3 and a ==3: #getting clother of the apple
            return self.reward[3]
        

    def distanceHeadApple(self):
        d=np.sqrt(np.square(self.snake.position[0]-self.foodSpawner.position[0])+np.square(self.snake.position[1]-self.foodSpawner.position[1]))
        if d >= self.distance:
            self.distance=d
            return 2
        else:
            self.distance=d
            return 3


    def displayFunc(self,score,game,epsilon):
        pygame.time.delay(1)
        pygame.display.set_caption('Snake - '+str(score)+' | Epoch - '+str(game)+' | Time - '+str(self.time)+' | Eps - '+str(epsilon))
        self.win.fill(pygame.Color(176,226,255))
        pygame.draw.rect(self.win,pygame.Color(0,0,0),pygame.Rect(10,10,self.size,self.size))
        for pos in self.snake.getBody(): 
                    pygame.draw.rect(self.win,pygame.Color(255,248,220),pygame.Rect(pos[0],pos[1],10,10))
        pygame.draw.rect(self.win,pygame.Color(220,20,60),pygame.Rect(self.foodPosition[0],self.foodPosition[1],10,10))
        pygame.display.flip()
        self.fps.tick(24)