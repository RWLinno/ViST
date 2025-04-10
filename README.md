# ViST: Vision-enhanced Spatio-Temporal Forecasting

This code is a PyTorch implementation of our paper "ViST".

## Requirements
We implement the experiments on a Linux Server with CUDA 12.2 equipped with 4x A6000 GPUs. For convenience, execute the following command.
```
# Install Python
conda create -n ViST python==3.11
conda activate ViST

# Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install other dependencies
pip install -r requirements.txt
```

## Dataset
You can download the all_data.zip file from [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp?usp=sharing). Unzip the files to the datasets/ directory:
refer to [BasicTS](https://github.com/zezhishao/BasicTS).
```
unzip /path/to/all_data.zip -d datasets/
```

### Quick Start
```bash
unset TQDM_DISABLE
python experiments/train.py -c ViST/largest/ViST_SD.py -g 0

export TQDM_DISABLE=1
nohup python experiments/train.py -c ViST/alation/ViSTv_SD.py --gpu '3' > logs/ViSTv_SD.txt 2>&1 &
```

## Tutorials
hide

## Citation
hide

## Acknowlegements
hide
