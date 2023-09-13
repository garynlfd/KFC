# KFC

KFC: Kinship Verification with Fair Contrastive loss and Multi-Task Learning

## News
**`2023-08`**: Accepted to [BMVC 2023](https://bmvc2023.org/)

## Introduction
KFC: Kinship Verification with Fair Contrastive Loss and Multi-Task Learning aims to solve the kinship verification task while improving racial fairness. We utilize an attention module in our model and employ multi-task learning to enhance accuracy in kinship verification. Additionally, we have developed a new fair contrastive loss function and use gradient reversal to decrease the standard deviation among the four races (African, Asian, Caucasian, Indian).

Furthermore, we combine six kinship datasets (see below) to create a large kinship dataset. Moreover, we have annotated every identity in our dataset to include racial information.

## Requirements
### Installation
1. Clone this project and create virtual environment
    ```bash
    $ git clone https://github.com/garynlfd/KFC.git
    $ conda create --name KFC python=3.8
    $ conda create activate KFC
    ```
2. Install requirements
    ```bash
    $ pip install -r requirements.txt
    ```
### Datasets
1. Please download these datasets on their websites.
+  CornellKin
    + http://chenlab.ece.cornell.edu/projects/KinshipVerification/
+  UBKinFace
    + http://www1.ece.neu.edu/~yunfu/research/Kinface/Kinface.htm
+  KinFaceW-I, KinFaceW-II
    + https://www.kinfacew.com/download.html
+  Family101
    + http://chenlab.ece.cornell.edu/projects/KinshipClassification/index.html
+  FIW
    + https://web.northeastern.edu/smilelab/fiw/
2. Please place these datasets in the same folder as train.py, find.py and test.py:
```text
./KFC
├── train.py
├── find.py
├── test.py
├── ...(other files)
├── Cornell_Kin/
├── UB_KinFace/
├── KinFaceW-I/
├── KinFaceW-II/
├── Family101_150x120/
├── Train(from FIW)/
├── Validation(from FIW)/
└── Test(from FIW)/
```
## Command
### Training
>>batch_size: default 25  
sample: The folder in which the data should be placed, corresponding to data files folder  
save_path: THe folder in which the ckpt will be saved, corresponding to log files folder  
epochs: default 100  
beta: temperature parameters default 0.08  
log_path: name the log file  
gpu: choose which gpu you want to use  
```
$ python train.py --batch_size 25 \
                --sample ./data_files \
                --save_path ./log_files \
                --epochs 100 --beta 0.08 \
                --log_path log_files/{file_name}.txt \
                --gpu 1
```
### Finding
>>sample: The folder in which the data should be placed, corresponding to data files folder  
save_path: THe folder in which the ckpt will be saved, corresponding to log files folder  
batch_size: default 50  
log_path: name the log file  
gpu: choose which gpu you want to use  
```
$ python find.py --sample ./data_files \
               --save_path ./log_files \
               --batch_size 50 \
               --log_path log_files/{file_name}.txt \
               --gpu 1
```

### Testing
>>sample: The folder in which the data should be placed, corresponding to data files folder  
save_path: THe folder in which the ckpt will be saved, corresponding to log files folder  
threshold: the output value from find.py  
batch_size: default 50  
log_path: name the log file  
gpu: choose which gpu you want to use  
```
$ python test.py --sample ./sample0 \
               --save_path ./log_files \
               --threshold {output value from find.py} \
               --batch_size 50 \
               --log_path log_files/{file_name}.txt \
               --gpu 1
```

## Acknowledgements
Our implementation uses code from the following repositories:
+ [RFIW 2021](https://github.com/Zhangxm520/rfiw2021/)