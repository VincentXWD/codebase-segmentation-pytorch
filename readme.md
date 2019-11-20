# codebase-segmentation-pytorch

*If you have any suggestions or questions, please start an issue or contact [me](https://github.com/VincentXWD) directly.*

[![Build Status](https://travis-ci.org/VincentXWD/codebase-segmentation-pytorch.svg?branch=master)](https://travis-ci.org/VincentXWD/codebase-segmentation-pytorch)

A codebase for segmentation tasks in pytorch. Goals:
- To provide an easy-in-use framework for image semantic segmentation researchers.
- To build a fast parallel computing framework utilizing features of newest version of PyTorch. Support not only single-node-multi-gpus but also multi-nodes-multi-gpus.
- To provide fine tuned & strong performance models as baseline for researchers.


## News
- [11/11/2019] Update pre-trained models: ResNet101-PSPNet on Cityscapes, Dilated ResNet50-PSPNet on Cityscapes. Click [HERE](#Performance).

- [05/11/2019] Support distributed training & evaluating.


## Pretrained model
We provide fine-tuned Networks with different backbone. We strongly recommend you to load the model by [eval.py](./eval.py). Also you can load it manually. Please note that the checkpoint has not only the model but also the optimizer information. You should load it use the key `checkpoint['state_dict']`.

The pretrained models are still updating.


## Requirements
- **python>=3.6**
- **opencv-python==3.4.2.17**
- torch>=1.0.10(better >= 1.3.0)
- imageio
- pandas
- medpy
- scipy
- pyyaml

## How to install

**There are two ways to install this framework:**
- According to the [requirements.txt](./requirements.txt).

  You can install all requirements according to the requirements.txt by the command below:
  ```
  pip install -r requirements.txt --user
  ```
- Using conda [yml](./environment.yml) configure.

  If you are a conda user, you can create yet another a new python environment by this command:
  ```
  conda env create -f environment.yml
  ```

- Please note that if you're using an old version of CUDA. Please modify the pytorch and torchvision version in requirements.txt but make sure the version of pytorch supports parallel computing.

## Quickstart

- To define a network:
```python
from models import PSPNet

net = PSPNet(
    encoder_name='resnet101',
    encoder_weights='imagenet',
    classes=19,
    auxiliary_loss=Your_Auxiliary_Loss,
    auxloss_weight=Your_Auxloss_Weight,
    criterion=Your_Criterion)
```

## Experiment results


### Cityscapes-19
Please note that all backbones are ResNet-101.
SS stands for Single-Scale and MS stands for Multi-Scale which has scales including [0.5, 0.75, 1.0, 1.25, 1.5, 1.75].
During training phase, UNet used batch size of 8 and sync BN.

|Model|meanIU(SS)|meanIU(MS)|
|----|----|----|
|UNet|68.403%|69.785%|


## Time consuming evaluation


### Training phase on Cityscapes-19
This codebase implements multiprocessing for training manually. The implementation document will be updated later.

We use 4 Nvidia GEFORCE GTX 1080 Tis to do this experiment. Setting batch_size=8, training 200 epochs on cityscapes-19 dataset. Experiment results will be updated soon.

|Model|Training time|
|----|----|
|Naive|≈27 hours|
|**Ours**|7 hours 44 minutes|


### Evaluating phase on Cityscapes-19

We also implement concurrent evaluating method.

We use 4 Nvidia GEFORCE GTX 1080 Tis to do this experiment. Here's comparison with naive implementation in evaluating phase.

|Model|meanIU(SS)|Time consuming(SS)|meanIU(MS)|Time consuming(MS)|
|----|----|----|----|----|
|Naive|68.403%|≈22 minutes|69.785%|≈4 hours|
|**Ours**|68.403%|≈7 minutes|69.785%|≈50 minutes|


### Performance

This section accumulated all models' performance implemented by this framework.

All models are trained in 200 epochs and with batch size equals 8. More details please refer the config files.

|Method|Backbone|Dataset|meanIU(SS)|meanIU(MS)|Pretrained Model|
|---|---|---|---|----|----|
|UNet|ResNet101|Cityscapes-19|70.221%|72.500%|None|
|PSPNet|ResNet101|Cityscapes-19|72.681%|74.863%|[Google Drive](https://drive.google.com/file/d/1yJD3SZvcPHXvfqjuYh2Bv_gthmN-WVEK/view?usp=sharing)|
|PSPNet|Dilated-ResNet50|Cityscapes-19|75.782%|76.579%|[Google Drive](https://drive.google.com/file/d/1Gph-t4zuJowP9cNYFmCuT1oKsg0onAQv/view?usp=sharing)|
|PSPNet|Dilated-ResNet101|Cityscapes-19|77.452%|78.585%|[Google Drive](https://drive.google.com/file/d/1Hee-hu0n686W_-Ck0NbFcfvTKkwLLReS/view?usp=sharing)|


## TODO(xwd): Finish documents & annotations.
