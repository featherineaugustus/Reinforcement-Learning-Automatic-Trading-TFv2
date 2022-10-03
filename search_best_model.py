# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:35:52 2022

@author: Featherine
"""


#%%############################################################################
# Import libraries
###############################################################################
# import os
# import math
# import random
# import numpy as np
import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
# from keras.models import load_model

# from tqdm import tqdm_notebook, tqdm
# from collections import deque


#%%############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    df = pd.read_csv('Results/Train/Results.csv')
    
    df = df.sort_values(['Total Profit', 'Net Worth'], ascending=False)
    # df = df.sort_values(['Net Worth'], ascending=False)
    df = df[(df['HOLD R'] > 0.1) & (df['BUY R'] > 0.1) & (df['SELL R'] > 0.1)]
    df = df[(df['Inventory Worth'] < 1000)]
    
    print(df[['Total Profit', 'Net Worth', 'Current Cash', 'Inventory Worth',
              'HOLD R', 'BUY R', 'SELL R']])
    
    results = pd.DataFrame(df[['Total Profit', 'Net Worth']])
    results.to_csv('Checkpoints/Best Model.csv')