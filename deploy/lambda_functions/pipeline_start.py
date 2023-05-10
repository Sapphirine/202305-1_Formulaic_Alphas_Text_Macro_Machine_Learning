#Starting a notebook instance
import boto3
import logging

def lambda_handler(event, context):
    client = boto3.client('sagemaker')
    client.start_notebook_instance(NotebookInstanceName='stock-bot-pipeline')
    return 0