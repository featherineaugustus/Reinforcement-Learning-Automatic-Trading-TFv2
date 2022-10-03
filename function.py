# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:53:41 2022

@author: Featherine
"""

#%%############################################################################
# Import libraries
###############################################################################
# import os
import math
# import random
import numpy as np
# import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

# from tqdm import tqdm_notebook, tqdm
# from collections import deque


#%%############################################################################
# Functions
###############################################################################
def sigmoid(x):
    return 1/(1 + math.exp(-x))
    

def stock_price_format(n):
    if n < 0:
        return '-$' + str(round(abs(n),1))
    else:
        return '$' + str(round(abs(n),1))
    
    
def dataset_loader(stock_name, train_test):
    '''
    Parameters
    ----------
    stock_name : str
        A string of a stock's name in capital letters.
    train_test : str
        Either 'Train' or 'Test', to indicate what the system is used for.
    
    Returns
    -------
    dataset : DataFrame
        The data containing different features over time.
    '''

    if train_test == 'Train':
        dataset = pdr.DataReader(stock_name, 
                                 data_source='yahoo',
                                 start='2020-01-01',
                                 end='2021-01-01',
                                 )
    else:
        dataset = pdr.DataReader(stock_name, 
                                 data_source='yahoo',
                                 start='2021-01-01',
                                 end='2022-01-01',
                                 ) 
        
    dataset.to_csv('Data/' + train_test + '/' + stock_name + '.csv', 
                   index=False)
    #start_date = str(dataset.index[0]).split()[0]
    #end_date = str(dataset.index[1]).split()[0]  
    # close = dataset['Close']
  
    return dataset
    

def state_creator(dataset, timestep, window_size):
    '''
    Parameters
    ----------
    dataset : DataFrame
        The dataset containing different features.
    timestep : int
        The timestep to predict.
    window_size : int
        The previous window_size to extract features from.

    Returns
    -------
   state : np.array
        The state of the current time step use to predict the action.
        It consist of the fraction change across each time step
    '''
  
    state = []
    
    # columns = dataset.columns
    columns = ['Close']
    
    for column in columns:
        data = dataset[column]
        starting_id = timestep - window_size + 1
      
        if starting_id >= 0:
            windowed_data = data[starting_id:timestep+1]
        else:
            windowed_data = -starting_id * [data[0]] + list(data[0:timestep+1])
        
        for i in range(window_size - 1):
            # state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))
            # diff = sigmoid(windowed_data[i+1] - windowed_data[i])
            diff = (windowed_data[i+1] - windowed_data[i])/windowed_data[i]
            state.append(diff)
    
    state = np.array([state]) 
    
    return state


def features_extraction(trader, current_price,
                        initial_cash, current_cash,
                        counter_hold, counter_buy, counter_sell,
                        total_hold, total_buy, total_sell
                        ):
    '''
    Parameters
    ----------
    trader : Class
        Trader.
    current_price : float
        The current price of the stock.
    initial_cash : float
        The initial amount of cash.
    current_cash : float
        The current amount of cash.
    counter_hold : int
        The current consecutive of hold action performed by the agent.
    counter_buy : int
        The current consecutive of buy action performed by the agent.
    counter_sell : int
        The current consecutive of sell action performed by the agent.
    total_hold : int
        The total count of hold action performed by the agent.
    total_buy : int
        The total count of buy action performed by the agent.
    total_sell : int
        The total count of sell action performed by the agent.

    Returns
    -------
    features : list
        A list of all the additional features that is to be appended to the
        state to increase the amount of useful features.
    '''
    
    total_all = total_hold + total_buy + total_sell
    # counter_all = counter_hold + counter_buy + counter_sell
    
    inventory = trader.inventory.copy()
    
    features = [
                # current_price, # Current price of the stock
                
                # Check trader inventory
                len(inventory), # How many is inside
                # sum(inventory), # The value inside
                # inventory[0] if len(inventory)>0 else 0, # Price of first unit
                # np.mean(inventory), # Average price of units
                
                # Price ratio/difference
                # inventory[0]/current_price if len(inventory)>0 else 0,
                # (inventory[0]-current_price)/current_price if len(inventory)>0 else 0,
                
                
                # We initalize the initial cash
                # initial_cash,  # Initial cash
                # current_cash,  # Current cash
                
                # We get the fix value as                
                counter_hold, 
                counter_buy, 
                counter_sell,

                # We get the fraction as the total trading time is not consistent                
                # counter_hold/counter_all if counter_all!=0 else 0, # Consecutive hold 
                # counter_buy/counter_all if counter_all!=0 else 0,  # Consecutive buy
                # counter_sell/counter_all if counter_all!=0 else 0, # Consecutive sell

                # We get the fraction as the total trading time is not consistent                
                total_hold/total_all if total_all!=0 else 0, # Total hold
                total_buy/total_all if total_all!=0 else 0,  # Total buy
                total_sell/total_all if total_all!=0 else 0, # Total sell
                ]
    
    return features



def plot_buy_sell(company, dataset, 
                  states_buy, states_sell,
                  networth, total_profit,
                  investment, train_test
                  ):
    '''
    Parameters
    ----------
    company : str
        The company name.
    dataset : DataFrame
        A dataframe containing all the information of the company.
    states_buy : list
        A list of t where buy occurs.
    states_sell : list
        A list of t where sell occurs.
    networth : float
        The current cash + the next worth of the inventory at current time.
    total_profit : float
        The gain from sales.
    investment : float
        The net gain percentage after trading.
    train_test : str
        Either 'Train' or 'Test', to indicate what the system is used for.

    Returns
    -------
    A figure of the price of the stock, and the location where
    buy and sell occurs
    None.
    '''

    data = dataset['Close']
    
    fig = plt.figure(figsize = (15,5))
    
    plt.plot(data, color='r', lw=2.)
    plt.plot(data, '^', markersize=10, color='m', label = 'buying signal', 
             markevery = states_buy)
    plt.plot(data, 'v', markersize=10, color='k', label = 'selling signal',
             markevery = states_sell)
    
    plt.title(company + ' - ' + 
              'NW: \$' + str(round(networth,3)) + ' - ' +
              'Total Gains: \$' + stock_price_format(total_profit) + ' - ' + 
              'Investment Rate: ' + stock_price_format(investment) + '% - ' + 
              'Buy: ' + str(len(states_buy)) + ' - ' + 
              'Sell: ' + str(len(states_sell)))
    plt.xlabel('Days')
    plt.ylabel('Price')
    
    plt.legend()
    plt.savefig('Results/' + train_test + '/' + company + '.png')
    plt.close('all')
    
    return None