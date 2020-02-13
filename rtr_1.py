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


### import surprise libraries
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import KNNBasic
from surprise import Dataset
from surprise.model_selection import GridSearchCV


### create Dataset
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(2.0, 10.0))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(ratings_df, reader)

# We'll use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

###one hot encode item df
one_hot_items = pd.get_dummies(item_df)
one_hot_items.head()

### trim Dataset
df['item_count'] = df.groupby(['item_id'])['item_id'].transform('count')
condition3 = df['item_count'] < 20
small_df = df[condition3]
### then get dummies for that Dataset

###cosine similarity

###other option -- limit to dresses only
dress_condition = ((df['category'] == 'dress') | (df['category'] == 'sheath') | (df['category'] == 'shirtdress') |
(df['category'] == 'shift') | (df['category'] == 'ballgown') | (df['category'] == 'frock') |
(df['category'] == 'kaftan') | (df['category'] == 'caftan') | (df['category'] == 'gown') | (df['category'] == 'print'))

dress_df = df[dress_condition]

dress_item_df = dress_df[['item_id', 'rating', 'rented_for', 'category']].copy()

dress_ratings_df = dress_df[['user_id', 'item_id', 'rating']]

###one hot encode item df
dress_items_dummies = pd.get_dummies(dress_item_df)
dress_items_dummies.head()

items_df_dress = dress_items_dummies.groupby(['item_id']).mean()
