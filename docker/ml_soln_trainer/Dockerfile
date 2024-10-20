# See tutorial for extending pre-built Sagemaker images:
#   https://docs.aws.amazon.com/sagemaker/latest/dg/prebuilt-containers-extend.html
# See readme for the sagemaker-training-toolkit framework used in the image
#   https://github.com/aws/sagemaker-training-toolkit/blob/master/README.md
# See the list of available base images:
#   https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# Source code for the chosen base image:
#   https://github.com/aws/deep-learning-containers/blob/master/tensorflow/training/docker/2.16/py3/cu123/Dockerfile.gpu
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.16.2-gpu-py310-cu123-ubuntu20.04-sagemaker

COPY --from=dist ml_solutions-1.0.0-py3-none-any.whl .

# Create a dummy tensorflow package to prevent pip from attempting to install it.
# This is because the image contains a custom AWS build of tensorflow, but the package is called tensorflow-gpu.
# When dependencies require tensorflow, pip does not realize that it is already present,
# so it attempts to install the open source version.
# See https://stackoverflow.com/a/76423135
RUN mkdir /usr/local/lib/python3.10/site-packages/tensorflow-2.16.2.dist-info
RUN touch /usr/local/lib/python3.10/site-packages/tensorflow-2.16.2.dist-info/METADATA

# Dummy torch package
RUN mkdir /usr/local/lib/python3.10/site-packages/torch-2.2.2.dist-info
RUN touch /usr/local/lib/python3.10/site-packages/torch-2.2.2.dist-info/METADATA

RUN pip install ml_solutions-1.0.0-py3-none-any.whl

COPY entrypoint.py /opt/ml/code/entrypoint.py

ENV SAGEMAKER_PROGRAM=entrypoint.py
