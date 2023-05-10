import io
import os

import boto3
import pandas as pd

import pickle

AWS_S3_BUCKET = "stock-bot-data"
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_KEY"]

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def write_to_s3(df, key, format = "csv"):
    with io.StringIO() as buffer:
        if format == "csv":
            df.to_csv(buffer, index=False)
        elif format == "feather":
            df.to_feather(buffer)

        response = s3_client.put_object(
            Bucket=AWS_S3_BUCKET, Key=key, Body=buffer.getvalue()
        )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")

def read_from_s3(key, format="csv"):
    response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=key)

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        body = response.get("Body")
        if format == "csv":
            return pd.read_csv(body)
        elif format == "excel":
            return pd.read_excel(io.BytesIO(body.read()))
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")

def list_s3_contents(key):
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=key)
    files = response.get("Contents")
    return files

def pickle_dump_to_s3(obj, key):
    pickle_byte_obj = pickle.dumps([obj]) 
    s3_resource = boto3.resource('s3')
    s3_resource.Object(AWS_S3_BUCKET, key).put(Body=pickle_byte_obj)