# Flow estimation by RAFT
The original code is taken from:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

The code are also quoted and modified from [OCLR](https://github.com/Jyxarthur/OCLR_model/) and [motiongrouping](https://github.com/charigyang/motiongrouping).

## Requirements
The original code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch
conda install matplotlib
conda install tensorboard
conda install scipy
conda install opencv
```

## Inference
Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing).

To estimate flows, set up dataset path in ```run_inference.py```, and then:
```Shell
python run_inference.py
```
where ```gap``` represents the frame gaps for flow estimations and ```reverse = [0, 1]``` corresponds to forward and backward flows, respectively.
