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

### change height from feet' inches" to inches (float)
condition = df['height'] != 'nan'
df = df[condition]

def parse_ht(ht):
    # format: 7' 0.0"
    ht_ = ht.split("' ")
    ft_ = float(ht_[0])
    in_ = float(ht_[1].replace("\"",""))
    return (12*ft_) + in_

df['height'] = df['height'].astype(str)
df['height'] = df['height'].str.split('"')
df['height'] = df['height'].apply(lambda x:x[0])
df['height'] = df['height'].apply(lambda x:parse_ht(x))

#### change weight column to drop 'lbs'
df['weight'] = df['weight'].str.split('\D+')
df['weight'] = df['weight'].apply(lambda x:x[0])
df['weight'] = df['weight'].astype('int32')

### remove outlier from "rented_for" columns
condition2 = df['rented_for'] != 'party: cocktail'
df = df[condition2]


##regroup categories (reduce from 68 to 7)
recat1 = df.replace(['dress', 'sheath', 'shirtdress', 'shift', 'ballgown', 'frock', 'kaftan', 'caftan', 'gown', 'print'], 'dresses')
recat2 = recat1.replace(['romper', 'jumpsuit', 'overalls', 'combo', 'suit'], 'jumpsuits')
recat3 = recat2.replace(['jogger', 'trousers', 'tight', 'jeans', 'sweatpants', 'leggings', 'pants', 'culottes', 'legging', 'pant', 'culotte', 'trouser'], 'pants')
recat4 = recat3.replace(['sweater', 'duster', 'cardigan', 'sweatshirt', 'pullover', 'turtleneck', 'hoodie', 'sweatershirt'], 'sweaters')
recat5 = recat4.replace(['jacket', 'coat', 'trench', 'cape', 'bomber', 'blazer', 'vest', 'poncho', 'down', 'parka', 'peacoat', 'overcoat'], 'outerwear')
recat6 = recat5.replace(['top,', 'shirt', 'blouse', 'tank', 'tunic', 'knit', 'tee', 'henley', 'blouson', 't-shirt', 'kimono', 'cami', 'crewneck', 'buttondown', 'for'], 'tops')
recat7 = recat6.replace(['mini', 'skirt', 'maxi','midi', 'skirts', 'skort'], 'skirts')
recat8 = recat7.replace(['top'], 'tops')

##rename recategorized df variable
df = recat8

#### split df into user_df and item_df
user_df = df[['user_id', 'bust_size', 'cup_size', 'body_type', 'weight', 'height', 'age']].copy()
item_df = df[['item_id', 'size', 'fit', 'rating', 'rented_for', 'category']].copy()
