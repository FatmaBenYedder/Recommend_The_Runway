import config
from bs4 import BeautifulSoup as BS
import mysql.connector
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import *
import time

df = pd.read_csv('test.csv', index_col=0)

## take spaces out of column names
df = df.rename(columns={'bust size': 'bust_size', 'rented for': 'rented_for', 'body type': 'body_type'})

## split up numbers and letters in bust_size
df['bust_size'] = df['bust_size'].str.split('(\D+)')
df['cup_size'] = df.bust_size.str[1]
df['bust_size'] = df.bust_size.str[0]
