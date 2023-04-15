# BMNet

CVPR 2022 oral (Official implementation of  [Bijective Mapping Network for Shadow Removal](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Bijective_Mapping_Network_for_Shadow_Removal_CVPR_2022_paper.pdf).

Yurui Zhu†, Jie Huang†, Xueyang Fu∗, Feng Zhao, Qibin Sun, Zheng-Jun Zha

†Equal Contributions
*Corresponding Author

University of Science and Technology of China (USTC)

## Introduction

This repository is the **official implementation** of the paper, "Bijective Mapping Network for Shadow Removal", where more implementation details are presented.

### 0. Hyper-Parameters setting

Overall, most parameters can be set in options/train/train_Enhance_ISTD.yml or options/train/train_Enhance_SRD.yml

### 1. Dataset Preparation

1) Download datasets and set the following structure
```
|-- ISTD_Dataset
    |-- train
        |-- train_A # shadow image
        |-- train_B # shadow mask
        |-- train_C # shadow-free GT
    |-- test
        |-- test_A # shadow image
        |-- test_B # shadow mask
        |-- test_C # shadow-free GT
```


2)  Create xx_train.txt and xx_test.txt files and put them  in `BMNet/MainNet/data/`.  

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

 You should modify the path of pre-training weights and run:

```python
python eval.py --opt options/test/test_Enhance_ISTD.yml or test_Enhance_SRD.yml or test_Enhance_AISTD.yml
```

## Dataset

ISTD dataset/SRD dataset/AISTD dataset

Please refer to previous project of shadow removal (see https://github.com/jinyeying/DC-ShadowNet-Hard-and-Soft-Shadow-Removal)

## Our results

Results on ISTD dataset (I have uploaded to https://drive.google.com/file/d/1cKRS26fgSOyIDqriD2fQFIcvyi2V8PIC/view?usp=sharing)

Results on SRD dataset (I have uploaded to https://drive.google.com/file/d/1Evi9-MWigJHuEwUov0w4v-gQqmZF1NPV/view?usp=sharing)

Results on AISTD dataset (I have uploaded to https://drive.google.com/file/d/1rg_hjihxIw4ypeQsiUavTWQ3dXD01qGu/view?usp=sharing)

##  Pre-trained Weights

The pre-trained weights (ISTD, AISTD and SRD) have been uploaded in `BMNet/MainNet/pretrain/`.  

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

