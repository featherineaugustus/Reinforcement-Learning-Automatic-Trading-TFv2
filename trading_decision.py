# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:35:11 2022

@author: Featherine
"""


#%%############################################################################
# Import libraries
###############################################################################
# import os
# import math
# import random
import numpy as np
import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
# from keras.models import load_model

# from tqdm import tqdm_notebook, tqdm
# from collections import deque


#%%############################################################################
# Import User Defined Class and Functions
###############################################################################
# from models import AI_Trader
from function import sigmoid
from function import stock_price_format
# from function import dataset_loader
from function import state_creator  
from function import features_extraction
from function import plot_buy_sell


#%%############################################################################
# Functions
###############################################################################
def trading_decision(dataset, company, 
                     trader, window_size, batch_size,
                     results, train_test,
                     episode=1):
    '''
    Parameters
    ----------
    dataset : DataFrame
        A dataframe containing all the information of the company.
    company : str
        The company name.
    trader : Class
        The class of the AITrader
    window_size : int
        The window length to backtrack in order to compute the state.
    batch_size : int
        The batch size used for training.
    results : list
        The result list.
    train_test : str
        Either 'Train' or 'Test', to indicate what the system is used for.
    episode : TYPE, optional
        The index used. The default is 1.

    Returns
    -------
    trader : Class
        The updated class of the AITrader.
    results : list
        The result list appended with the latest results
    '''
    
    action_dict = {0: 'HOLD',
                   1: 'BUY',
                   2: 'SELL'}
    
    # Initial Values
    stop_buy_limit = int(window_size/2)                 # Day left to stop buying
    state = state_creator(dataset, 0, window_size + 1)  # Initial state
    initial_cash = 10000                                # Initial cash
    current_cash = initial_cash                         # Current cash
    total_profit = 0                                    # Total profit from sales
    trader.inventory = []                               # Inventory  
    current_price = dataset['Close'][0]                 # Current price of the unit
    
    # Counters
    # Total count of hold, buy, sell
    total_hold, total_buy, total_sell = 0, 0, 0         
    # Consecutive count of hold, buy, sell
    counter_hold, counter_buy, counter_sell = 0, 0, 0
    total_reward = 0
    
    # Log -> Save outputs
    log = []
    states_buy = []
    states_sell = []
    
    # Compute state (Closing price + Features)
    state = state_creator(dataset, 0, window_size + 1)
    features = features_extraction(trader, current_price,
                                    initial_cash, current_cash,
                                    counter_hold, counter_buy, counter_sell,
                                    total_hold, total_buy, total_sell
                                    )
    state = np.array([list(np.append(state, features))])
    
    # Looping through the time series
    for t in range(len(dataset) - 1):
        
        current_price = dataset['Close'][t] # Get current price
        action = trader.trade(state)        # Get current action
        reward = 0                          # Initialize reward to 0
        
        action_true = action
        
        #######################################################################
        # Buy
        #######################################################################
        if ((action == 1) and 
            (current_cash >= current_price) and 
            (t <= len(dataset) - 1 - stop_buy_limit)
            ):
            counter_buy +=1
            total_buy +=1
            
            # If there is something in the inventory
            if len(trader.inventory) > 0:
                # If the latest stock is more expensive than the current price
                # Buy is good
                if trader.inventory[0] > current_price:
                    reward = 1.0
            # If the inventory is empty, buying is good
            else:
                reward = 1.0
                
            trader.inventory.append(current_price) # Buy stock
            current_cash -= current_price          # Spent some cash
            states_buy.append(t)                   # Keep track of buy state


            print('Day ' + str(t+1) + ': BUY:  ' + 
                  stock_price_format(current_price) + 
                  ' - Inventory: ' + str(len(trader.inventory)) + 
                  ' - Reward: ' + str(reward))
            
        #######################################################################
        # Sell
        #######################################################################
        elif ((action == 2) and 
              (len(trader.inventory) > 0)
              ):
            counter_sell += 1
            total_sell += 1
            
            buy_price = trader.inventory.pop(0) # Sell the latest product
            profit = current_price - buy_price  # Compute reward (profit)
            total_profit += profit              # Total profit
            current_cash += current_price       # Put back cash 
            states_sell.append(t)               # Keep track of sell state
            
            # if profit > 0:
            #     reward = round(np.log2(profit+1), 1)
            # elif profit < 0:
            #     reward = round(-np.log2(abs(profit+1)), 1)
            # elif profit == 0:
            #     reward = 0
            if profit > 0:
                reward = np.max([1, round(profit/5,1)])
            else:
                reward = -np.max([1, round(abs(profit)/5,1)])
                
            temp = 'Profit' if (reward > 0) else 'Loss'
            print('Day ' + str(t+1) + ': SELL: ' + 
                  stock_price_format(current_price) + ' - ' + 
                  temp + ': ' + stock_price_format(profit) + 
                  ' - Reward: ' + str(reward))       
        
        #######################################################################
        # Hold
        #######################################################################
        elif (action == 0):
            counter_hold += 1         # Number of consecutive hold
            total_hold += 1           # Total hold in the entire run
            # reward = -counter_hold # -x reward if hold (should buy or sell)
            # reward = -0.5               # -1 reward if hold (should buy or sell)
            print('Day ' + str(t+1) + ': HOLD' + 
                  ' - Inventory: ' + str(len(trader.inventory)) + 
                  ' - Reward: ' + str(reward))
            
        else:
            # Update action as it is either 
            # 1) Try to buy but no money to buy
            # 2) Try to sell but no inventory to sell
            # Hence, cannot do anything, but should not penalize the robot
            counter_hold += 1   # Number of consecutive hold
            total_hold += 1     # Total hold in the entire run
            reward = -1
            print('Day ' + str(t+1) + ': HOLD as ' + action_dict[action] + 
                  ' cannot be completed ' + 
                  ' - Inventory: ' + str(len(trader.inventory)) + 
                  # '- Current Cash: ' + stock_price_format(current_cash) + 
                  ' - Reward: ' + str(reward))
            action_true = 0          

        
        # Reset counter
        if action == 0:
            counter_hold = 0
        elif action == 1:
            counter_buy = 0
        elif action == 2:
            counter_sell = 0
            
        
        total_reward = total_reward + reward
        
        #######################################################################
        # Check if the forecasting ends
        #######################################################################
        done = True if (t == len(dataset) - 1) else False
          
        #######################################################################
        # Compute next state
        #######################################################################
        state_next = state_creator(dataset, t+1, window_size + 1)
        features = features_extraction(trader, current_price,
                                       initial_cash, current_cash,
                                       counter_hold, counter_buy, counter_sell,
                                       total_hold, total_buy, total_sell
                                       )
        state_next = np.array([list(np.append(state_next, features))])
        
        #######################################################################
        # Upload data into trader memory
        #######################################################################
        # reward = 2*(sigmoid(reward) - 0.5)
        trader.memory.append((state, action, reward, state_next, done))
        state = state_next # Update state
        

        #######################################################################
        # Train model
        #######################################################################
        if ((len(trader.memory) > batch_size) and # If have enough memory to train
            (t%10 == 0) and                       # Train once every 10 step
            (train_test == 'Train')):             # Only during training mode
            trader.batch_train()
        
        
        #######################################################################
        # Check networth
        #######################################################################
        inventory_worth = len(trader.inventory)*current_price
        networth = inventory_worth + current_cash
        investment = (networth - initial_cash)/initial_cash*100
        
        
        #######################################################################
        # Upload log
        #######################################################################
        total_all = total_hold + total_buy + total_sell
        
        log.append([t+1, current_price, action_true, action, action_dict[action],
                    trader.inventory.copy(), reward, total_reward,
   
                    initial_cash, current_cash,
                    inventory_worth, networth,
                    total_profit, investment,
                    total_hold, total_buy, total_sell, total_all,
                    total_hold/total_all if total_all!=0 else 0, # Total hold
                    total_buy/total_all if total_all!=0 else 0,  # Total buy
                    total_sell/total_all if total_all!=0 else 0, # Total sell
                    ]
                   )
        
    ###########################################################################
    # Save
    ###########################################################################
    print('############################################')
    print('TOTAL PROFIT : ' + stock_price_format(total_profit))
    print('TOTAL CASH   : ' + stock_price_format(current_cash))
    print('INVENTORY    : ' + stock_price_format(inventory_worth))
    print('NET WORTH    : ' + stock_price_format(networth))
    print('TOTAL REWARD : ' + str(round(total_reward,1)))
    print('############################################')
    
    ###########################################################################
    # Save Results
    ###########################################################################
    results.append([episode, 
                    
                    total_reward,
                    initial_cash, current_cash,
                    inventory_worth, networth,
                    total_profit, investment,
                    total_hold, total_buy, total_sell, total_all,
                    total_hold/total_all if total_all!=0 else 0, # Total hold
                    total_buy/total_all if total_all!=0 else 0,  # Total buy
                    total_sell/total_all if total_all!=0 else 0, # Total sell
                    ])
    
    columns = ['Epoch',
               
               'Total Reward',
               'Initial Cash', 'Current Cash',
               'Inventory Worth', 'Net Worth',
               'Total Profit', 'Investment',
               'HOLD', 'BUY', 'SELL', 'ALL',
               'HOLD R', 'BUY R', 'SELL R',]
    
    results_save = pd.DataFrame(results, columns = columns)
    results_save.to_csv('Results/' + train_test + 
                        '/Results.csv', index=False)
  
    ###############################################################
    # Save Log
    ###############################################################
    columns = ['Day', 'Price', 'Action True', 'Action Final', 'Action',
               'Inventory', 'Reward', 'Total Reward',
               
               'Initial Cash', 'Current Cash',
               'Inventory Worth', 'Net Worth',
               'Total Profit', 'Investment',
               'HOLD', 'BUY', 'SELL', 'ALL',
               'HOLD R', 'BUY R', 'SELL R',]
    
    log_save = pd.DataFrame(log, columns = columns)
    log_save.to_csv('Log/' + train_test + 
                    '/Log ' + str(episode) + '.csv', index=False)
        
    #######################################################################
    # Plot results
    #######################################################################   
    plot_buy_sell(company, dataset, 
                  states_buy, states_sell,
                  networth, total_profit,
                  investment, train_test)
    
    #######################################################################
    # Save Model
    #######################################################################
    if train_test == 'Train':
        trader.model.save('Checkpoints/' + train_test + 
                          '/ai_trader_{}.h5'.format(episode))
        
    return trader, results