# See tutorial for extending pre-built Sagemaker images:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/prebuilt-containers-extend.html
# See readme for the sagemaker-training-toolkit framework used in the image
#   https://github.com/aws/sagemaker-training-toolkit/blob/master/README.md
# See the list of available base images:
#   https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# Source code for the chosen base image:
#   https://github.com/aws/deep-learning-containers/blob/master/tensorflow/training/docker/2.16/py3/cu123/Dockerfile.gpu
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.16.2-gpu-py310-cu123-ubuntu20.04-sagemaker

COPY --from=dist kaggle_solutions-1.0.0-py3-none-any.whl .

RUN pip install kaggle_solutions-1.0.0-py3-none-any.whl

COPY entrypoint.py /opt/ml/code/entrypoint.py

ENV SAGEMAKER_PROGRAM entrypoint.py
