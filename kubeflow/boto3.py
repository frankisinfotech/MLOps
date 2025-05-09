import boto3

s3 = boto3.resource('s3')

def get_bucket():
    for bucket in s3.buckets.all():
        print(bucket.name)
get_bucket()

#------------------------
# Download Files from S3
#------------------------
bucket_name = 'test-mlops-bkt'  # Replace with your bucket name
file_key = 'model/rental_prediction_model.pkl'  # Replace with your folder location on S3
local_file_path = 'MyNB.pkl'
#local_file_path = 'local_rental_prediction_mode.pkl'  # Replace with your local path

def download_file(bucket_name, file_key, local_file_path):

        s3.Bucket(bucket_name).download_file(file_key, local_file_path)
        print(f"Downloaded {file_key} from {bucket_name} to {local_file_path}")

download_file(bucket_name, file_key, local_file_path)

#--------------------
# Upload Files to S3
#--------------------
bucket_name = 'test-mlops-bkt'  # Replace with your bucket name
file_key = 'data/MyNB.pkl'  # Replace with your folder location on s3
local_file_path = 'MyNB.pkl'  # Replace with your local path

def upload_file(bucket_name, file_key, local_file_path):
    
        s3.Bucket(bucket_name).upload_file(local_file_path, file_key)
        print(f"Uploaded {local_file_path} to {bucket_name}/{file_key}")

upload_file(bucket_name, file_key, local_file_path)
print("File Upload successfully.")
