# Library Imports
import kfp
from kfp import dsl
import logging
import os

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

    bucket = "iriscloudbt-mlapp"
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


# Main execution
if __name__ == "__main__":
    # kfp.compiler.Compiler().compile(rental_prediction_pipeline, 'rental_prediction_pipeline.yaml')

    kfp_endpoint = None
    client = kfp.Client(host=kfp_endpoint)

    # Experiment name
    experiment_name = 'Rental Prediction Pipeline Experiment'

    # List all experiments
    experiments = client.list_experiments()

    # Search for the experiment by display_name
    experiment = next((exp for exp in experiments.experiments if exp.display_name == experiment_name), None)

    if experiment:
        # If the experiment exists, fetch its experiment_id
        experiment_id = experiment.experiment_id
        logging.info(f"Found experiment: {experiment_name}, Experiment ID: {experiment_id}")
    else:
        # If the experiment does not exist, create it
        logging.info(f"Experiment '{experiment_name}' not found. Creating a new experiment.")
        experiment = client.create_experiment(experiment_name)
        experiment_id = experiment.experiment_id
        logging.info(f"Created new experiment. Experiment ID: {experiment_id}")

    # List and delete the previous run if it exists
    list_runs = client.list_runs(experiment_id=experiment_id)
    if list_runs.runs:
        previous_run_id = list_runs.runs[0].run_id
        logging.info(f"Deleting previous run: {previous_run_id}")
        # Delete the previous run
        client.delete_run(previous_run_id)
    else:
        logging.info("No previous runs found to delete.")

        
    try:
        client.create_run_from_pipeline_func(
            rental_prediction_pipeline,
            experiment_name=experiment_name,
            enable_caching=False
        )
        logging.info("Pipeline run initiated")
    except Exception as e:
        logging.error(f"Failed to create run from pipeline function: {e}")
