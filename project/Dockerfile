FROM continuumio/miniconda3:4.10.3p1
WORKDIR /project
RUN conda install \
    tensorflow \
    tensorflow-io \
    scikit-learn \
    scikit-image \
    opencv-python \
    imagecodecs \
    matplotlib \
    jupyterlab
COPY ./README.md /project
COPY ./dataTraining /project/dataTraining
COPY ./notebooks /project/notebooks
COPY ./utils /project/utils
CMD ["jupyter-lab","--ip=0.0.0.0","--no-browser","--allow-root"]