# Human Detection and Segmentation via Multi-view Consensus 
This is the source code for the ICCV'21 paper "Human Detection and Segmentation via Multi-view Consensus".

## Dependencies
<li> Python 3.6
<li> PyTorch 1.4
<li> Cuda 9.2
  
## Training
  Change the dataset path in the files dataset/dataset_factory.py and dataset/SkiPTZ.py. You can start training with the following command:
```
$ cd src
$ python configs/run_config.py configs/config_dict_mvc_ski.py
```
 
