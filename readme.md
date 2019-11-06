# codebase-segmentation-pytorch

My personal codebase for segmentation tasks in pytorch.

Support distributed training now.

## Requirements
- **python>=3.6**
- **opencv-python==3.4.2.17**
- torch>=1.0.10
- imageio
- pandas
- medpy
- scipy
- pyyaml


## How to install
**There are two ways to install this framework:**
* **According to the [requirements.txt](./requirements.txt).**

  You can install all requirements according to the requirements.txt by the command below:
  ```
  pip install -r requirements.txt
  ```
* **Using conda [yml](./environment.yml) configure.**

  If you are a conda user, you can create yet another a new python environment by this command:
  ```
  conda env create -f environment.yml
  ```


## Experiment results
### **Cityscapes-19**
Please note that all backbones are Dilated ResNet-101.
SS stands for Single-Scale and MS stands for Multi-Scale which has scales including [0.5, 0.75, 1.0, 1.25, 1.5, 1.75].
During training phase, UNet used batch size of 8 and sync BN.

|Model|meanIU(SS)|meanIU(MS)|
|----|----|----|
|UNet|68.403%|69.785%|

## Time consuming evaluation

### **Training phase on Cityscapes-19**
This codebase implements multiprocessing for training manually. The implementation document will be updated later.

We use 4 Nvidia GEFORCE GTX 1080 Tis to do this experiment. Setting batch_size=8, training 200 epochs on cityscapes-19 dataset. Experiment results will be updated soon.

|Model|Training time|
|----|----|----|----|
|Naive|≈27 hours|
|**Ours**|7 hours 44 minutes|

### **Evaluating phase on Cityscapes-19**

We also implement concurrent evaluating method.

We use 4 Nvidia GEFORCE GTX 1080 Tis to do this experiment. Here's comparison with naive implementation in evaluating phase.

|Model|meanIU(SS)|Time consuming(SS)|meanIU(MS)|Time consuming(MS)|
|----|----|----|----|----|
|Naive|68.403%|≈22 minutes|69.785%|≈4 hours|
|**Ours**|68.403%|≈7 minutes|69.785%|≈50 minutes|


### **Performance of models**

This section accumulated all models' performance implemented by this framework.

All models are trained in 200 epochs and with batch size equals 8. More details please refer the config files.

|Method|Backbone|Dataset|meanIU(SS)|meanIU(MS)|
|---|---|---|---|----|
|UNet|Dilated-ResNet101|Cityscapes-19|70.221%|72.500%|



### TODO(xwd): Finish documents & annotations.
