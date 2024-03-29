## Multi-Agent Perception via learnable token demo
<img src="teaser/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This is the PyTorch implementation and modification based on When2com: <br>
**When2com: Multi-Agent Perception via Communication Graph Grouping**<br>
[__***Yen-Cheng Liu***__](https://ycliu93.github.io/), [Junjiao Tian](https://www.linkedin.com/in/junjiao-tian-42b9758a/), [Nathaniel Glaser](https://sites.google.com/view/nathanglaser/), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/)<br>
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020 <br>


[[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_When2com_Multi-Agent_Perception_via_Communication_Graph_Grouping_CVPR_2020_paper.pdf)] [[GitHub](https://github.gatech.edu/RIPL/multi-agent-perception)] [[Project](https://ycliu93.github.io/projects/multi-agent-perception.html)]

<p align="center">
<img src="teaser/1359-teaser.gif" width="50%">
</p>

## Prerequisites
- Python 3.6
- Pytorch 0.4.1
- Other required packages in `requirement.txt`


## Getting started
### Download and install miniconda 
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Create conda environment
```
conda create -n semseg python=3.6
source actviate semseg
```

### Install the required packages
```
pip install -r requirements.txt
```

### Download AirSim-MAP dataset and unzip it. 
- Download the zip file you would like to run

[![Alt text](https://ycliu93.github.io/projects/cvpr20_assets/airsim_map.png)](https://gtvault-my.sharepoint.com/:f:/g/personal/yliu3133_gatech_edu/Ett0G1_5YYdBpgojk0uWESgBi95dO79LkbYaKRhlBIkVJQ?e=vdjklb/)


### Move the datasets to the dataset path
```
mkdir dataset
mv (dataset folder name) dataset/
```

### Training 
```
# [Single-request multi-support] All norm  
python train.py --config configs/srms_allnorm.yml --gpu=0

# [Multi-request multi-support] when2com model  
python train.py --config configs/mrms_when2com.yml --gpu=0

# [Multi-request multi-support] Modified when2com model with learnable token
python train.py --config configs/mrms_MIMO2.yml --gpu=0

```

### Testing 
```
# [Single-request multi-support] All norm  
python test.py --config configs/srms-allnorm.yml --model_path <your trained weights> --gpu=0

# [Multi-request multi-support] when2com model  
python test.py --config configs/mrms-when2com.yml --model_path <your trained weights> --gpu=0
```
