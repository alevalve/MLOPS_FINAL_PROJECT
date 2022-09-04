import Batch 
import pandas as pd

Batch.main

from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)

def test_read_data():
    data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    categorical = ['PULocationID', 'DOLocationID']
    columns = ['PULocationID', 'DOLocationID','trip_distance','fare_amount','passenger_count','tpep_pickup_datetime','tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df_actual = Batch.prepare_data(df, categorical)

    data_expected = [
        ('-1', '-1', 8.0),
        ('1','1', 8.0),
    ]

    columns_test = ['PULocationID', 'DOLocationID','duration']
    df_expected = pd.DataFrame(data_expected, columns=columns_test)
    print(df_expected)

    assert(df_actual['PULocationID'] == df_expected['PULocationID']).all()
    assert(df_actual['DULocationID'] == df_expected['DULocationID']).all()
    print(df_actual['duration'] == df_expected['duration']).abs().sum() < 0.0001


    

    


