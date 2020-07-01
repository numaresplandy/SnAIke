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
    def __init__(self,size,run_max_time):
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


    def reset(self):
        self.snake=Snake(self.size)
        self.foodSpawner=FoodSpawner(self.size,self.snake.getBody())
        self.foodPosition=self.foodSpawner.SpawnFood(self.snake.getBody())
        self.currentState=self.getState(self.snake.getHeadPos(),self.snake.getBody(),self.foodSpawner)
        self.nextState=[]
        self.distance=0
        self.time=0
        return self.currentState

    def step(self,action):
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
            reward = self.giveReward(self.distanceHeadApple())
        self.next_state=self.getState(self.snake.getHeadPos(),self.snake.getBody(),self.foodSpawner)
        return self.next_state, reward, done
    
    def updateStates(self): 
        self.currentState=self.nextState
        self.nextState=[]

    
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
        

    def getApplePosition(self,posHead,foodspw):
        apple = ''
        #posHead = self.snake.position
        posFood = foodspw.getFoodPos()
        if posFood[0]==posHead[0] and posFood[1] in range(0,posHead[1]):
            #NORTH
            apple='N'
        elif posFood[0]==posHead[0] and posFood[1] in range(posHead[1],(self.size)+10): 
            #SOUTH 
            apple='S' 
        elif posFood[1]==posHead[1] and posFood[0] in range(0,posHead[0]): 
            #WEST
            apple='W'
        elif posFood[1]==posHead[1] and posFood[0] in range(posHead[0],(self.size)+10): 
            #EAST
            apple='E'
        elif posFood[1] in range(0,posHead[1]) and posFood[0] in range(0,posHead[0]): 
            #NORT-WEST
            apple='NW' 
        elif  posFood[1] in range(0,posHead[1]) and posFood[0] in range(posHead[0],(self.size)+10): 
            #NORT-EAST
            apple='NE'
        elif  posFood[1] in range(posHead[1],(self.size)+10) and posFood[0] in range(0,posHead[0]): 
            #SOUTH-WEST
            apple='SW'
        elif  posFood[1] in range(posHead[1],(self.size)+10) and posFood[0] in range(posHead[0],(self.size)+10): 
            #SOUTH-EAST
            apple='SE'

        return self.apple.index(apple)
        #return apple 


    def giveReward(self,ID):
        if ID==0: #touch the apple
            return 30
        if ID==1:  #touch a wall or itself
            return -100 
        if ID==2: #getting far away of the apple
            return -10
        if ID==3: #getting clother of the apple
            return 10


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