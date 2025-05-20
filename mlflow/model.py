# Import Libraries
from mlflow.models import infer_signature
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np
import logging
import os
import boto3
import shutil

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

bucket = "iriscliriscloudbt-mlapp"
data_s3_path = "data/rental_1000.csv"
local_data_path = "data/rental_1000.csv"

local_model_path = "model/model.pkl"
model_s3_path = "model/model.pkl"

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

# Load the dataset into a Pandas DataFrame
# Create data folder if it doesn't exist
os.makedirs(os.path.dirname(local_data_path), exist_ok=True)

# Download the dataset from S3
s3_client.download_file(bucket, data_s3_path, local_data_path)

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(local_data_path)

# Define features (X) and labels (y)
X = df[['rooms', 'sqft']].values
y = df['price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Initialize the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Define a function to calculate evaluation metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return rmse

# Evaluate the model
rmse = eval_metrics(y_test, y_pred)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://localhost:5050")

# Create a new MLflow Experiment
mlflow.set_experiment("rental-prediction-experiment")

# Log the loss metric
mlflow.log_metric("rmse", rmse)

# Infer the model signature
signature = infer_signature(X_train, lr.predict(X_train))

# Log the model
model_info = mlflow.sklearn.log_model(
    sk_model=lr,
    artifact_path="rental_prediction_model",
    signature=signature,
    registered_model_name="rental_prediction_model",
)

# Remove the existing directory if it exists
if os.path.exists("data"):
    shutil.rmtree("data")

if os.path.exists("model"):
    shutil.rmtree("model")

# Create the model directory if it doesn't exist
mlflow.sklearn.save_model(lr, "model")

# Upload the model to S3
s3_client.upload_file(local_model_path, bucket, model_s3_path)
