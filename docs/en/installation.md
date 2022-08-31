## Installation
XRLocalization is mainly based on Python >= 3.7, XRPrimer >= 0.5.2 and PyTorch >= 1.1.
As long as the environment is ready, the installation is very simple. In this section,
we show how to prepare an environment and install XRLocalization.

### Ubuntu
### Prerequisites
If you have install docker on your machine, we recommend that you directly
use docker to build the running environment.
```commandline
cd docker
docker build . -t xrlocalization:latest
```
If not, you can refer the following steps.

**step 0.**
Install the dependencies from the default Ubuntu repositories:
```commandline
sudo apt-get update && apt-get install -y \
        build-essential \
        libgoogle-glog-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libatlas-base-dev \
        libsuitesparse-dev
```

**step 1.**
Download and install Miniconda from [official website](https://docs.conda.io/en/latest/miniconda.html).
For example(on Linux 640-bit machine):
```commandline
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
sh Miniconda3-py37_4.12.0-Linux-x86_64.sh
```

**step 2.**
Create an environment with Python=3.7.
```commandline
conda create --name xrloc python=3.7
conda activate xrloc
```
Note that Python >= 3.7 is required.

**step 3.**
Install XRPrimer
```commandline
pip install xrprimer==0.5.2 -i \
    https://repo.sensetime.com/repository/pypi/simple
```

**step 4.**
Install PyTorch

Please refer to [here](https://pytorch.org/) for PyTorch installation. For example:
```commandline
conda install pytorch torchvision -c pytorch
```
Note that we only test our code with Pytorch 1.1 and Pytorch 1.9.


**step 5.**
Clone xrlocalization
```commandline
clone  --recursive  https://gitlab.bj.sensetime.com/openxrlab/xrlocalization
```
Or
```commandline
clone  https://gitlab.bj.sensetime.com/openxrlab/xrlocalization
git submodule update --init
```

### Installation
```commandline
cd xrlocalization
python3 setup.py install
```
