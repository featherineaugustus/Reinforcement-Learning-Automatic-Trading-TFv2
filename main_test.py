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
# import random
# import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
from keras.models import load_model

# from tqdm import tqdm_notebook, tqdm
# from collections import deque

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# tf.debugging.set_log_device_placement(True)


#%%############################################################################
# Import User Defined Class and Functions
###############################################################################
from models import AI_Trader
# from function import sigmoid
# from function import stock_price_format
from function import dataset_loader
# from function import state_creator  
# from function import features_extraction
# from function import plot_buy_sell
from trading_decision import trading_decision


#%%############################################################################
# Main
###############################################################################
if __name__ == "__main__":   
    # Initialize parameters
    window_size = 10
    episodes = 100
    batch_size = 32
    # additional_feature = window_size*5 + 13
    # additional_feature = window_size*5
    additional_feature = 0
    
    # Testing, no training
    train_test = 'Test'
    
    # Find best model
    best_model = pd.read_csv('Checkpoints/Best Model.csv')
    best_index = best_model['Unnamed: 0'][0]
    
    # Load data and model, no training
    trader = AI_Trader(state_size = window_size + additional_feature,
                       batch_size = batch_size)
    trader.model = load_model('Checkpoints/Train/ai_trader_{}.h5'.format(2))

    # Select dataset to test
    company_list = ['TSLA', 'MSFT', 'AMZN',
                    'AAPL','GOOG','FB', 
                    'AMD', 'ADSK', 'BLK',
                    'HPQ', 'IBM', 'JPM',
                    'META', 'MU', 'NVDA',
                    'PYPL', 'TWTR', 'USB',
                    'V'
                    ]
    
    results = []
    ###########################################################################
    # Testing
    ###########################################################################
    for company in company_list:
    
        print('\nCompany: ' + company)
        dataset = dataset_loader(company, train_test)
        # data.to_csv('Data/' + company  + '.csv',index=False)
      
        trader, results = trading_decision(dataset, company, 
                                           trader, window_size, batch_size,
                                           results, train_test,
                                           episode = company)