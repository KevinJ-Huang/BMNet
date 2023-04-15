# BMNet

CVPR 2022 (Official implementation of "Bijective Mapping Network for Shadow Removal")

Yurui Zhu†, Jie Huang†, Xueyang Fu∗, Feng Zhao, Qibin Sun, Zheng-Jun Zha

†Equal Contributions
*Corresponding Author

University of Science and Technology of China (USTC)

## Introduction

This repository is the **official implementation** of the paper, "Bijective Mapping Network for Shadow Removal", where more implementation details are presented.

### 0. Hyper-Parameters setting

Overall, most parameters can be set in options/train/train_Enhance_ISTD.yml or options/train/train_Enhance_SRD.yml

### 1. Dataset Preparation

Create a .txt file to put the path of the dataset using 

```python
python create_txt.py
```

### 2. Training

Firstly, we need to step into ColorTrans folder and train the subnetwork for color map restoration:

```python
python train.py --opt options/train/train_Enhance.yml
```

Then, save the .pth file and put the file path to the ConditionNet in the Enhance_arch.py in MainNet folder.

Next, you can train the shadow removal network as:

```python
python train.py --opt options/train/train_Enhance_ISTD.yml or train_Enhance_SRD.yml or train_Enhance_ISTD.yml
```


### 3. Inference

```python
python test.py --opt options/train/test_Enhance_ISTD.yml or test_Enhance_SRD.yml or test_Enhance_ISTD.yml
```

## Dataset

ISTD dataset/SRD dataset/AISTD dataset

Please refer to previous project of shadow removal (see https://github.com/jinyeying/DC-ShadowNet-Hard-and-Soft-Shadow-Removal)

## Our results

Results on ISTD dataset (I have uploaded to https://drive.google.com/file/d/1cKRS26fgSOyIDqriD2fQFIcvyi2V8PIC/view?usp=sharing)

Results on SRD dataset (I have uploaded to https://drive.google.com/file/d/1Evi9-MWigJHuEwUov0w4v-gQqmZF1NPV/view?usp=sharing)

Results on AISTD dataset (I have uploaded to https://drive.google.com/file/d/1rg_hjihxIw4ypeQsiUavTWQ3dXD01qGu/view?usp=sharing)

## Pre-trained Weights



## Contact

If you have any problem with the released code, please do not hesitate to contact me by email (zyr@mail.ustc.edu.cn or hj0117@mail.ustc.edu.cn).

## Cite

```
@InProceedings{Zhu_2022_CVPR,
    author    = {Zhu, Yurui and Huang, Jie and Fu, Xueyang and Zhao, Feng and Sun, Qibin and Zha, Zheng-Jun},
    title     = {Bijective Mapping Network for Shadow Removal},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5627-5636}
```

