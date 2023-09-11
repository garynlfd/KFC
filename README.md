# KFC

KFC: Kinship Verification with Fair Contrastive loss and Multi-Task Learning

## News
**`2023-08`**: Accepted to [BMVC 2023](https://bmvc2023.org/)

## Introduction
KFC: Kinship Verification with Fair Contrastive loss and Multi-Task Learning, aim to solve kinship verification task while improving racial fairness. We utilize attention module in our model and multi-task learning to enhance accuracy in kinship verification while creating a new fair contrastive loss function and using gradient reversal to deacrease standard deviation among 4 races(African, Asian, Caucasian, Indian).
We also combine 6 kinship datasets(see below) to get a large kinship dataset. Moreover, we annotate every identity in our dataset to gain racial information.

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