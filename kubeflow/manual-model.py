# Library Imports
import kfp
from kfp import dsl
import logging
import os

# Fetch the AWS keys from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')


@dsl.component(packages_to_install=["numpy", "pandas", "scikit-learn", "boto3"])
def build_model(aws_access_key_id: str, aws_secret_access_key: str):
    # Import Libraries
    import boto3
    # from botocore.exceptions import ClientError

    import pandas as pd
    import pickle

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    # from sklearn.metrics iawsmport mean_squared_error

    bucket = "model-mlapp-bkt"
    data_s3_path = "data/rental_1000.csv"
    model_s3_path = "model/rental_prediction_model.pkl"
    local_data_path = "rental_1000.csv"
    local_model_path = "rental_prediction_model.pkl"

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Download data from S3
    s3_client.download_file(bucket, data_s3_path, local_data_path)

    # Load the dataset
    rentalDF = pd.read_csv(local_data_path)

    # Prepare the features and labels
    X = rentalDF[["rooms", "sqft"]].values
    y = rentalDF["price"].values
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    # Train the model
    lr = LinearRegression()
    model = lr.fit(X_train, y_train)

    # Save the model using pickle
    with open(local_model_path, 'wb') as f:
        pickle.dump(model, f)

    # Upload the model to S3
    s3_client.upload_file(local_model_path, bucket, model_s3_path)


@dsl.pipeline
def rental_prediction_pipeline():
    build_model(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

from kfp import compiler

compiler.Compiler().compile(rental_prediction_pipeline, 'rental_prediction_pipeline.yaml')

#client = kfp.Client(host=None)
#client.create_run_from_pipeline_func(
    #rental_prediction_pipeline
    #)%
