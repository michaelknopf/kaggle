import json
import os
import sys

import boto3
import pandas as pd
import sagemaker
from keras.api.utils import to_categorical
from sagemaker.remote_function import remote, CustomFileFilter
from sklearn.model_selection import train_test_split


with open('./private/aws_creds.json') as f:
    aws_creds = json.load(f)

boto_session = boto3.Session(
    aws_access_key_id=aws_creds['key'],
    aws_secret_access_key=aws_creds['secret'],
    region_name='us-west-2',
)
sagemaker_session = sagemaker.session.Session(boto_session=boto_session)


# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# image = '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.16.2-gpu-py310-cu123-ubuntu20.04-sagemaker'
# instance = 'ml.g4dn.xlarge'
@remote(
    image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.16.2-gpu-py310-cu123-ubuntu20.04-sagemaker',
    instance_type='ml.m5.xlarge',
    # instance_type='ml.g4dn.xlarge',
    # dependencies=''
    job_name_prefix='mnist',
    role='arn:aws:iam::394547497655:role/service-role/AmazonSageMaker-ExecutionRole-20240306T211248',
    s3_root_uri='s3://sagemaker-us-west-2-394547497655/remote-function/mnist',
    sagemaker_session=sagemaker_session,
    include_local_workdir=True,
    custom_file_filter=CustomFileFilter(
        ignore_name_patterns=[
            ".venv/*"
            "*.ipynb",
            "data",
        ]
    )
)
def train_remote(X_train, X_val, Y_train, Y_val):
    sys.path.append('./packages')
    sys.path.append(f'{os.getcwd()}/packages')
    print(f"""
    Current working directory: {os.getcwd()}
    System path: {sys.path}
    List dir: {os.listdir()}
    """)
    from digit_recognizer.train_remote import train
    return train(X_train, X_val, Y_train, Y_val)


def main():
    # Load the data
    train = pd.read_csv("./packages/digit_recognizer/data/kaggle_dataset/train.csv")
    test = pd.read_csv("./packages/digit_recognizer/data/kaggle_dataset/test.csv")

    Y_train = train["label"]

    # Drop 'label' column
    X_train: pd.DataFrame = train.drop(labels=["label"], axis=1)

    # free some space
    del train

    # Normalize the data
    X_train = X_train / 255.0
    test = test / 255.0

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_train = to_categorical(Y_train, num_classes=10)

    # Set the random seed
    random_seed = 2

    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train,
        Y_train,
        train_size=1000,
        test_size=100,
        # test_size=0.1,
        random_state=random_seed
    )

    # train in sagemaker
    history = train_remote(X_train, X_val, Y_train, Y_val)

if __name__ == '__main__':
    main()
