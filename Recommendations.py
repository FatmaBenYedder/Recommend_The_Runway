#import packages
:
# import libraries
import numpy as np
import pandas as pd

from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split

# importing relevant libraries
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import SVD
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline
from surprise.model_selection import GridSearchCV

#drop NANs
df = pd.read_csv('updated_df.csv')
df = df.dropna()

#drop review date
df = df.drop(columns = 'review_date')

## take spaces out of column names
df = df.rename(columns={'bust size': 'bust_size', 'rented for': 'rented_for', 'body type': 'body_type'})

## split up numbers and letters in bust_size
df['bust_size'] = df['bust_size'].str.split('(\D+)')
df['cup_size'] = df.bust_size.str[1]
df['bust_size'] = df.bust_size.str[0]

#parse height

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

#remove string from weight, convert to integer
df['weight'] = df['weight'].str.replace('lbs', '').astype(int)

#plot binned weights
plt.hist(df['weight'], bins=25)

#convert size to integer
df['size'] = df['size'].astype(int)

#create conditions
condition1 = df['user_id'] == 45337
condition2 = df['rented_for'] != 'party: cocktail'
df = df[condition2]

#reduce number of categories from 68 to 7 by grouping similar types of items
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

dress_condition = ((df['category'] == 'dress') | (df['category'] == 'sheath') | (df['category'] == 'shirtdress') |
(df['category'] == 'shift') | (df['category'] == 'ballgown') | (df['category'] == 'frock') |
(df['category'] == 'kaftan') | (df['category'] == 'caftan') | (df['category'] == 'gown') | (df['category'] == 'print'))

cols = ['user_id', 'item_id', 'rating']
small_df = df[cols]

dress_df = df[dress_condition]
user_df = df[['user_id', 'bust_size', 'cup_size', 'body_type', 'weight', 'height', 'age']].copy()
item_df = df[['item_id', 'rating', 'rented_for', 'category']].copy()
dress_item_df = dress_df[['item_id', 'rating', 'rented_for', 'category']].copy()
ratings_df = small_df[['user_id', 'item_id', 'rating']]
dress_ratings_df = dress_df[['user_id', 'item_id', 'rating']]

#drop review data
feats = list(df.columns.drop( ['review_summary', 'review_text']))

#one-hot-encode user dataframe
user_feats = list(user_df.columns)
user_ohe = user_df[user_feats]
user_ohe = pd.get_dummies(user_ohe, drop_first=True)

#begin rating prediction
reader = Reader(rating_scale=(2, 10))
data = Dataset.load_from_df(small_df[['user_id', 'item_id', 'rating']],reader)

dataset = data.build_full_trainset()
print('Number of users: ', dataset.n_users, '\n')
print('Number of items: ', dataset.n_items)

train, test = train_test_split(data, test_size=.2)

## Perform a gridsearch with SVD
# This cell may take several minutes to run
params = {'n_factors': [20, 50, 100],
         'reg_all': [0.02, 0.05, 0.1]}
g_s_svd = GridSearchCV(SVD,param_grid=params,n_jobs=-1)
g_s_svd.fit(data)

print(g_s_svd.best_score)
print(g_s_svd.best_params)

#begin recommendation engine
df['item_count'] = df.groupby(['item_id'])['item_id'].transform('count')

#create recommendations for items rented fewer than 25 times
condition3 = df['item_count'] < 25
small_df = df[condition3]

cols = ['user_id', 'item_id', 'rating']
rec_df = small_df[cols]

rec_df = rec_df.dropna()

reader = Reader(rating_scale=(2, 10))
data = Dataset.load_from_df(rec_df[['user_id', 'item_id', 'rating']],reader)

dataset = data.build_full_trainset()
print('Number of users: ', dataset.n_users, '\n')
print('Number of items: ', dataset.n_items)
train, test = train_test_split(data, test_size=.2)

#perform gridsearch with SVD
params = {'n_factors': [20, 50, 100],
         'reg_all': [0.02, 0.05, 0.1]}
g_s_svd = GridSearchCV(SVD,param_grid=params,n_jobs=-1)
g_s_svd.fit(data)
print(g_s_svd.best_score)
print(g_s_svd.best_params)

#begin KNN basic model
knn_basic = KNNBasic(sim_options={'name':'pearson', 'user_based':True})
cv_knn_basic = cross_validate(knn_basic, data, n_jobs=-1)

for i in cv_knn_basic.items():
    print(i)
print('-----------------------')
print(np.mean(cv_knn_basic['test_rmse']))

# cross validating with KNNBaseline
knn_baseline = KNNBaseline(sim_options={'name':'pearson', 'user_based':True})
cv_knn_baseline = cross_validate(knn_baseline,data)

for i in cv_knn_baseline.items():
    print(i)

np.mean(cv_knn_baseline['test_rmse'])

svd = SVD(n_factors= 50, reg_all=0.05)
svd.fit(dataset)

svd.predict(2, 4)
rec_df
