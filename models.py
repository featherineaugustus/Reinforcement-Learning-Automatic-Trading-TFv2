# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:44:20 2022

Deep Reinforcement Learning for Trading with TensorFlow 2.0

https://www.mlq.ai/deep-reinforcement-learning-for-trading-with-tensorflow-2-0/

@author: WeiYanPEH
"""

#%%############################################################################
# Import libraries
###############################################################################
# import os
# import math
import random
import numpy as np
# import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr

# from tqdm import tqdm_notebook, tqdm
from collections import deque

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# tf.debugging.set_log_device_placement(True)


#%%############################################################################
# Define User Defined Class and Functions
###############################################################################

class AI_Trader():
  
    def __init__(self, 
               state_size, 
               action_space=3, 
               model_name='AI_Trader',
               batch_size = 32,
               ):
    
        self.state_size = state_size 
        self.action_space = action_space # 3: Buy, Sell, Hold
        self.memory = deque(maxlen=1000) # Memory/experience
        self.inventory = []
        self.model_name = model_name
        
        self.alpha = 0.05
        self.gamma = 0.95 # Learning rate 
        self.epsilon = 1.0 # Explore rate
        self.epsilon_final = 0.05 # Minimum explore rate
        self.epsilon_decay = 0.999 # Explore decay
        self.batch_size = batch_size
        self.model = self.model_builder()
        
        self.state = None
        self.action = None
        self.reward = None
        self.state_next = None
    
        self.q_values = None
        self.q_values_next = None
        
    
    def model_builder(self):
          
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(self.state_size)))
        
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        
        # model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        # model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        # model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        # model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        # model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='softmax'))
        
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
        
        return model
    
    
    def trade(self, state):
        if random.random() <= self.epsilon:
            actions = random.randrange(self.action_space)
        else:
            actions = np.argmax(self.model.predict(state)[0])        
        return actions
    
    
    def batch_train(self):
    
        batch = []
        for i in range(len(self.memory) - self.batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])
    
        X_train = []
        y_train = []
    
        for state, action, reward, state_next, done in batch:

            self.state = state
            self.action = action
            self.reward = reward
            self.state_next = state_next

            # estimate q-values based on current state
            self.q_values = self.model.predict(state)
            self.q_values_next = self.model.predict(state_next)
            
            # print(self.q_values)
            # print(self.q_values_next)
            
            target = reward

            if not done:
                ## For linear activation function -> Output is huge 1000-5000
                # target = reward + self.gamma * np.amax(self.q_values_next[0])
                
                ## For softmax activation function -> Output is small 0-1
                target = target + reward * self.alpha * np.amax(self.q_values_next[0])
     
            # update the target for current action based on discounted reward
            self.q_values[0][action] = target
     
            X_train = state
            y_train = self.q_values
            
            self.model.fit(X_train, y_train, epochs=1, verbose=0)
    
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
        
