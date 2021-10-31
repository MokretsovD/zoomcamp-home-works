#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

# parameters

alpha = 1
n_splits = 5
output_file = 'model.bin'
file_name = 'autoscout24-germany-dataset.csv'

# data preparation based on EDA in notebook

print('Started data preparation')

df = pd.read_csv(file_name)

print('Number of Rows loaded: %d' % len(df))

# Make column names uniform
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Make non numerical values in dataframe uniform
string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Remove rows with empty model or gear
df = df[(df.model.isnull() == False) & (df.gear.isnull() == False)]

# Fill empty HP with mean
hp_mean = df.hp.mean()
df['hp'] = df['hp'].fillna(hp_mean)

# Remove duplicates
df = df.drop_duplicates(keep='first')    

# Remove price outliers, applying log transformation
df = df[df.price <= 450000]
df.price = np.log1p(df.price)

# Remove mileage outliers
df = df[df.mileage <= 400000]

# Remove hp outliers
df = df[(df.hp <= 400) & (df.hp > 30)]

# Add age
df['age'] = datetime.now().year - df['year']

# Delete offertype
del df['offertype']

# Fix fuel
df.fuel = df.fuel.replace('-/-_(fuel)', 'others')

print('Number of Rows after data cleaning: %d' % len(df))

print('Splitting the model')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

numerical = ['mileage', 'hp', 'year', 'age']
categorical = ['make', 'model', 'fuel', 'gear']

# training

print('Training the model')

def train(df_t, y_t):
    dicts = df_t[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = Ridge(alpha=alpha, random_state=42, normalize=False)
    model.fit(X_train, y_t)
    
    return dv, model


def predict(df,dv,model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict(X)
    
    return y_pred

def rmse(yp, y_predp):
    se = (yp - y_predp) ** 2
    mse = se.mean()
    return np.sqrt(mse)

# validation

print(f'Doing validation with alpha={alpha}')

kfold = KFold(n_splits=n_splits,shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.price.values
    y_val = df_val.price.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val,dv, model)

    rmse_val = rmse(y_val, y_pred)
    scores.append(rmse_val)

    fold = fold + 1
    
    print(f'rmse on fold {fold} is {rmse_val}')

print('Validation results:')
print('alpha=%s %.3f +- %.3f' % (alpha, np.mean(scores), np.std(scores)))

# training the final model

print('Training the final model')

dv, model = train(df_full_train, df_full_train.price.values)
y_pred = predict(df_test, dv, model)

y_test = df_test.price.values
rmse_val = rmse(y_test, y_pred)

print(f'rmse={rmse_val}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


print(f'The model is saved to {output_file}')