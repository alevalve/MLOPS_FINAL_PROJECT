import pickle
import pandas as pd
import numpy as np
import sklearn as sk
import datetime as dt
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pyarrow 
import fastparquet

year = 2022
month = 3

with open('lin_reg.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID','trip_distance','fare_amount','passenger_count','tpep_pickup_datetime','tpep_dropoff_datetime']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(f'/Users/agvg/Documents/DS/MLOPS/MLOPS/Final_Project/yellow_tripdata_{year}-{month}.parquet')

dicts = df[categorical].to_dict(orient='records')
Xval = dv.transform(dicts)
y_pred = lr.predict(Xval)
mean_duration = y_pred.mean()
print(mean_duration)