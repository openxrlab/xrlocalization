FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y gnupg

RUN apt-get install -y \
    build-essential \
    libgoogle-glog-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libatlas-base-dev \
    libsuitesparse-dev

RUN apt-get install -y wget

# Python dependency
RUN pip install xrprimer==0.5.2
RUN pip install scipy==1.6.1
RUN pip install opencv-python==3.4.9.33
RUN pip install matplotlib==3.2.1
RUN pip install scikit-learn==0.24.1
RUN pip install flask
RUN pip install h5py
RUN pip install pathlib
