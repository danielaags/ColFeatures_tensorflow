FROM python:3.7

WORKDIR /project

COPY requirements.txt project/requirements.txt
COPY dataTraining /project/dataTraining
COPY notebooks /project/notebooks
COPY utils /project/utils

RUN python -m pip install numpy
RUN python -m pip install matplotlib
RUN python -m pip install tensorflow
RUN python -m pip install tensorflow-io
RUN python -m pip install scikit-learn
RUN python -m pip install scikit-image
RUN python -m pip install opencv-python-headless
RUN python -m pip install imagecodecs
RUN python -m pip install jupyterlab