# KFC

KFC: Kinship Verification with Fair Contrastive loss and Multi-Task Learning

## News
**`2023-08`**: Accepted to [BMVC 2023](https://bmvc2023.org/)

## Introduction
KFC: Kinship Verification with Fair Contrastive Loss and Multi-Task Learning aims to solve the kinship verification task while improving racial fairness. We utilize an attention module in our model and employ multi-task learning to enhance accuracy in kinship verification. Additionally, we have developed a new fair contrastive loss function and use gradient reversal to decrease the standard deviation among the four races (African, Asian, Caucasian, Indian).

Furthermore, we combine six kinship datasets (see below) to create a large kinship dataset. Moreover, we have annotated every identity in our dataset to include racial information.

## Requirements
### Installation
### Datasets
Please download these datasets on their websites.
1. CornellKin
    + http://chenlab.ece.cornell.edu/projects/KinshipVerification/
2. UBKinFace
    + http://www1.ece.neu.edu/~yunfu/research/Kinface/Kinface.htm
3. KinFaceW-I, KinFaceW-II
    + https://www.kinfacew.com/download.html
4. Family101
    + http://chenlab.ece.cornell.edu/projects/KinshipClassification/index.html
5. FIW
    + https://web.northeastern.edu/smilelab/fiw/