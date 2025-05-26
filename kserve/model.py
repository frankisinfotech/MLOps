# Import Libraries
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import pandas as pd
import numpy as np
import logging
import boto3
import os
import shutil

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# AWS S3 upload details
bucket = "sigma-bucket-us-east-2"  # S3 bucket name
s3_prefix = 'rental-prediction-model'  # S3 key prefix for folder structure

local_data_path = "data/rental_1000.csv"
local_model_path = "rental-prediction-model"
data_s3_path = "data/rental_1000.csv"
model_s3_path = "rental-prediction-model"

# Fetch the AWS keys from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Check if the environment variables are set
if aws_access_key_id and aws_secret_access_key:
    logging.info("AWS Access Key and Secret Key have been retrieved successfully.")
    logging.info("AWS Access Key ID: %s", aws_access_key_id)
    logging.info("AWS Secret Access Key: %s",
        aws_secret_access_key[:4] + "*" * 16 + aws_secret_access_key[-4:])
else:
    raise EnvironmentError("AWS Access Key or Secret Key not set properly.")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Initialize the S3 client
s3_client = boto3.client('s3')

# Function to upload files to S3
def upload_to_s3(local_path, bucket, s3_key):
    try:
        s3_client.upload_file(local_path, bucket, s3_key)
        print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload {local_path} to S3: {e}")

# Function to upload an entire directory to S3
def upload_directory_to_s3(local_directory, bucket, s3_prefix):
    for root, dirs, files in os.walk(local_directory):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            
            # Create the relative S3 key by removing the local directory path from the file path
            relative_path = os.path.relpath(local_file_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # Ensure proper path formatting for S3

            # Upload the file
            upload_to_s3(local_file_path, bucket, s3_key)

# Create data folder if it doesn't exist
os.makedirs(os.path.dirname(local_data_path), exist_ok=True)

# Download the dataset from S3
s3_client.download_file(bucket, data_s3_path, local_data_path)

# Load the dataset
df = pd.read_csv(local_data_path)  # Ensure the dataset exists and the path is correct

# Features and Labels as DataFrame
X = df[['rooms', 'sqft']].values  
y = df['price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Build and train the model using pandas DataFrame for X
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Make predictions on the test set (using a DataFrame for the test data)
y_pred = model.predict(X_test)

# Evaluation function for logging metrics
def eval_metrics(pred, actual):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    return rmse,r2

# Set tracking URI for MLflow server
mlflow.set_tracking_uri('http://localhost:5050')  # Ensure your MLflow server is running

# Set the Experiment (Create or reuse)
mlflow.set_experiment("rental-prediction-experiment")

runs = mlflow.search_runs(order_by=["start_time desc"])  # Get most recent run
if not runs.empty:
    previous_run_id = runs.iloc[0].run_id
    mlflow.delete_run(previous_run_id)
    print(f"Deleted previous run with ID: {previous_run_id}")
else:
    print("No previous runs found to delete.")

model_signature = infer_signature(X_train, y_train)# Start a new MLflow run (this will create a new run each time the script is executed)
with mlflow.start_run() as run:
    # Log the model using feature names
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="rental-prediction-model",
        registered_model_name="rental-prediction-model",
        signature=model_signature,
    )
    
    # Log evaluation metrics (RMSE and R²)
    rmse, r2 = eval_metrics(y_pred, y_test)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Get the new run ID for the current run
    run_id = run.info.run_id
    print(f"Created new run with ID: {run_id}")

# Remove the existing directory if it exists
if os.path.exists("data"):
    shutil.rmtree("data")

# Remove the existing directory if it exists
if os.path.exists("rental-prediction-model"):
    shutil.rmtree("rental-prediction-model")

# Save the model locally before uploading to S3
local_model_path = "rental-prediction-model"

# Create the directory if it doesn't exist
os.makedirs(local_model_path, exist_ok=True)
mlflow.sklearn.save_model(model, local_model_path)

# Example usage:
upload_directory_to_s3(local_model_path, bucket, s3_prefix)
