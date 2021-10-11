# Human Detection and Segmentation via Multi-view Consensus 
This is the source code for the ICCV'21 paper "Human Detection and Segmentation via Multi-view Consensus".

## Dependencies
<li> Python 3.6
<li> PyTorch 1.4
<li> Cuda 9.2
  
## Training
  - Change the dataset path in the files dataset/dataset_factory.py and dataset/SkiPTZ.py. 
  - Download the [pre-trained models](https://drive.google.com/drive/folders/1oeY6SQwMwXiQJReDv-5dTyZcp_WBPofj?usp=sharing).
  - Create a folder named pretrained in the cloned directory and put the pre-trained models inside.
  
  You can start training with the following command:
```
$ cd src
$ python configs/run_config.py configs/config_dict_mvc_ski.py
```
 
