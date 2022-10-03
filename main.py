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
import os
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model

from tqdm import tqdm_notebook, tqdm
from collections import deque

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# tf.debugging.set_log_device_placement(True)

# Suppress TPU messages which start with "Executing op"

# import sys, re, logging

# class Filter(object):
#     def __init__(self, stream):
#         self.stream = stream

#     def __getattr__(self, attr_name):
#         return getattr(self.stream, attr_name)

#     def write(self, data):
#         if not data.startswith("Executing op") or  not data.startswith("2022-09-30"):
#             self.stream.write(data)
#             self.stream.flush()

#     def flush(self):
#         self.stream.flush()
  
# sys.stdout = Filter(sys.stdout)
# sys.stderr = Filter(sys.stderr)

# logger = logging.getLogger(__name__)

#%%############################################################################
# Import User Defined Class and Functions
###############################################################################
from models import AI_Trader
from function import sigmoid
from function import stock_price_format
from function import dataset_loader
from function import state_creator  
from function import features_extraction
from function import plot_buy_sell
from trading_decision import trading_decision


#%%############################################################################
# Create Folders
###############################################################################
if not os.path.exists('Data'):
    os.makedirs('Data/Train')
    os.makedirs('Data/Test')
    
if not os.path.exists('Results'):
    os.makedirs('Results/Train')
    os.makedirs('Results/Test')
    
if not os.path.exists('Log'):
    os.makedirs('Log/Train')
    os.makedirs('Log/Test')
    
if not os.path.exists('Checkpoints'):
    os.makedirs('Checkpoints/Train')
    os.makedirs('Checkpoints/Test')


#%%############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    # Initialize parameters
    window_size = 30
    episodes = 50
    batch_size = 32
    # additional_feature = window_size*5 + 13
    # additional_feature = window_size*5
    additional_feature = 7
    
    # Training
    train_test = 'Train'
 
    # Load model, prepare to train
    trader = AI_Trader(state_size = window_size + additional_feature,
                       batch_size = batch_size)
    trader.model.summary()
    
    # Select dataset to train
    company = 'ADSK'
    dataset = dataset_loader(company, train_test)
    # data.to_csv('Data/' + company + '.csv',index=False)
    
    results = []
    
    ###########################################################################
    # Training
    ###########################################################################
    for episode in range(1, episodes + 1):
        print('\nEpisode: {}/{}'.format(episode, episodes))
        trader, results = trading_decision(dataset, company, 
                                           trader, window_size, batch_size,
                                           results, train_test,
                                           episode)