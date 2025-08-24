from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import pickle
import os
import urllib.request

from airflow import DAG
from airflow.operators.python import PythonOperator

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-student',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize DAG
dag = DAG(
    'nyc_taxi_training_pipeline',
    default_args=default_args,
    description='NYC Taxi Training Pipeline with MLflow',
    schedule=None,
    catchup=False,
    tags=['mlops', 'training'],
)

# Task 1: Download and load data
def load_data(**context):
    """
    Question 3: Load March 2023 Yellow taxi trips data
    """
    data_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
    local_path = "/tmp/yellow_tripdata_2023-03.parquet"
    
    if not os.path.exists(local_path):
        print(f"Downloading data from {data_url}")
        urllib.request.urlretrieve(data_url, local_path)
        print("Download complete!")
    else:
        print("Data already exists, using cached version")
    
    print("Loading parquet file...")
    df = pd.read_parquet(local_path)
    
    print("="*50)
    print(f"QUESTION 3 ANSWER: {len(df):,} records loaded")
    print("="*50)
    
    df.to_parquet('/tmp/raw_data.parquet')
    
    return len(df)

# Task 2: Prepare data
def prepare_data(**context):
    """
    Question 4: Data preparation
    """
    def read_dataframe(filename):
        df = pd.read_parquet(filename)
        
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
        
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        
        return df
    
    print("Preparing data...")
    df = read_dataframe('/tmp/raw_data.parquet')
    
    print("="*50)
    print(f"QUESTION 4 ANSWER: {len(df):,} records after preparation")
    print("="*50)
    
    df.to_parquet('/tmp/prepared_data.parquet')
    
    return len(df)

# Task 3: Train model
def train_model(**context):
    """
    Question 5: Train a linear regression model
    """
    print("Training model...")
    df = pd.read_parquet('/tmp/prepared_data.parquet')
    
    categorical = ['PULocationID', 'DOLocationID']
    
    n = len(df)
    n_train = int(0.8 * n)
    
    df_train = df[:n_train]
    df_val = df[n_train:]
    
    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")

    train_dicts = df_train[categorical].to_dict(orient='records')
    val_dicts = df_val[categorical].to_dict(orient='records')
    
    print("Fitting DictVectorizer...")
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    
    print(f"Feature matrix shape: {X_train.shape}")

    y_train = df_train.duration.values
    y_val = df_val.duration.values

    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    print("="*50)
    print(f"QUESTION 5 ANSWER: Model intercept = {lr.intercept_:.2f}")
    print("="*50)
    
    y_pred_train = lr.predict(X_train)
    y_pred_val = lr.predict(X_val)
    
    # Using np.sqrt manually instead of squared=False parameter
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    
    print(f"RMSE on training: {rmse_train:.2f}")
    print(f"RMSE on validation: {rmse_val:.2f}")
    
    with open('/tmp/model.pkl', 'wb') as f:
        pickle.dump(lr, f)
    
    with open('/tmp/dv.pkl', 'wb') as f:
        pickle.dump(dv, f)
    
    return {
        'intercept': float(lr.intercept_),
        'rmse_train': float(rmse_train),
        'rmse_val': float(rmse_val)
    }

# Task 4: Register model with MLflow
def register_model_mlflow(**context):
    """
    Question 6: Register the model with MLflow
    """
    print("Registering model with MLflow...")
    
    mlflow.set_tracking_uri("sqlite:////tmp/mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID,DOLocationID")
        
        # Load model and vectorizer
        with open('/tmp/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('/tmp/dv.pkl', 'rb') as f:
            dv = pickle.load(f)
        
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="nyc-taxi-regressor"
        )
        
        mlflow.log_artifact('/tmp/dv.pkl', "preprocessor")
        
        run_id = mlflow.active_run().info.run_id

        try:
            model_uri = f"runs:/{run_id}/model"
            model_path = mlflow.artifacts.download_artifacts(model_uri)

            mlmodel_path = os.path.join(model_path, "MLmodel")
            if os.path.exists(mlmodel_path):
                with open(mlmodel_path, 'r') as f:
                    content = f.read()
                    print("="*50)
                    print("QUESTION 6: Checking MLModel file for model_size_bytes")
                    for line in content.split('\n'):
                        if 'model_size_bytes' in line:
                            print(f"Found: {line.strip()}")
                    print("="*50)
            
            model_pkl_path = os.path.join(model_path, "model.pkl")
            if os.path.exists(model_pkl_path):
                model_size = os.path.getsize(model_pkl_path)
                print(f"Actual model.pkl size: {model_size} bytes")
                print("="*50)
                print(f"QUESTION 6 ANSWER: {model_size} bytes")
                print("="*50)
        except Exception as e:
            print(f"Error checking model size: {e}")
        
        print(f"Model logged with run_id: {run_id}")
    
    return run_id

# Task 5: Print summary
def print_summary(**context):
    """
    Print summary of all answers
    """
    print("="*60)
    print("HOMEWORK ANSWERS SUMMARY")
    print("="*60)
    print("Question 1: Apache Airflow")
    print("Question 2: 3.0.4'")
    print("Question 3: 3,403,766 records (check load_data task logs)")
    print("Question 4: 3,316,216 records (check prepare_data task logs)")
    print("Question 5: 24.38 (check train_model task logs)")
    print("Question 6: 4,526 bytes (check register_model_mlflow task logs)")
    print("="*60)

task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

task_prepare_data = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    dag=dag,
)

task_train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

task_register_model = PythonOperator(
    task_id='register_model_mlflow',
    python_callable=register_model_mlflow,
    dag=dag,
)

task_summary = PythonOperator(
    task_id='print_summary',
    python_callable=print_summary,
    dag=dag,
)

# Set task dependencies
task_load_data >> task_prepare_data >> task_train_model >> task_register_model >> task_summary
