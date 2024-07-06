# AAAI 2024 STCNet
Motion Deblurring  via Spatial-Temporal Collaboration of Frames and Events
# Installation

The model is built in PyTorch 1.8.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these instructions

    conda create -n pytorch1 python=3.7
    conda activate pytorch1
    conda install pytorch=1.8 torchvision=0.3 cudatoolkit=9.0 -c pytorch
    pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm

Install warmup scheduler

    cd pytorch-gradual-warmup-lr; python setup.py install; cd ..

# Training and Evaluation
## Train
- Download the [GoPro events train dataset](https://pan.baidu.com/s/1lw-CW3QH-ZJdpP0CT9oMnw) and [GoPro events test dataset](https://pan.baidu.com/s/1UKV-sPGo9mRf7XJjZDoF7Q) (code: kmaz) to ./Datasets
- Train the model with default arguments by running

  python main_train.py

## Evaluation
- Download the [GoPro events test dataset](https://pan.baidu.com/s/1UKV-sPGo9mRf7XJjZDoF7Q) (code: kmaz) to ./Datasets
- Download the  [pretrained model](https://pan.baidu.com/s/193vCnygNkXT_GOq6PhRrhg) (code: svbb) to ./checkpoints/models/STCNet
- Test the model with default arguments by running

  python main_test.py
