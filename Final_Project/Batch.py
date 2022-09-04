import sys
import pickle
import pandas as pd

def prepare_data(df,categorical):
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df



def read_data(filename, categorical):
        df = pd.read_parquet(filename)
        return prepare_data(df, categorical)
        
    
def main(year, month):

    input_file = f'/Users/agvg/Documents/DS/MLOPS/MLOPS/Final_Project/yellow_tripdata_{year}-{month}.parquet'
    output_file = f'/Users/agvg/Documents/DS/MLOPS/MLOPS/Final_Project/yellow_tripdata_predictions_{year}-{month}.parquet'


    with open('lin_reg.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    categorical = ['PULocationID', 'DOLocationID','trip_distance','fare_amount','passenger_count','tpep_pickup_datetime','tpep_dropoff_datetime']


    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, engine='pyarrow', index=False)


if __name__ == '__main__':

    year = int(sys.argv[1])
    month = int(sys.argv[2])   
    main(year, month)
    


