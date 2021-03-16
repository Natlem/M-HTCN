A Pytorch Implementation for ICCV submission 9456

## Introduction
Please follow [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0) and [HTCN](https://github.com/chaoqichen/HTCN) respository to setup the environment. In this project, we use Pytorch 1.6.1 and CUDA version is 10.2

## Additional Information
 - We use Sacred([link](https://sacred.readthedocs.io/en/stable/)) to run/manage our experiments. It can be installed with:
 ```
 pip install sacred
 ```
 - To monitor the training, we used MongoObserver of Sacred with Omniboard. However, it is disabled by default. If you want to plug in your own monitoring. It can be done in **class LoggerForSacred** of **experiments/exp_utils.py** which will be called at every 100 iteration. By default, log are printed to stdout.

## Datasets
### Datasets Preparation
* **Cityscape and FoggyCityscape:** Download the [Cityscape](https://www.cityscapes-dataset.com/) dataset, see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data).
* **PASCAL_VOC 07+12:** Please follow the [instruction](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC dataset.
* **Clipart - Watercolor - Comic:** Please follow the [instruction](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets) to download

### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  
If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in ```lib/datasets/```, and add it to ```lib/datasets/factory.py```, ```lib/datasets/config_dataset.py```. Then, add the dataset option to ```lib/model/utils/parser_func.py```.

For RainCityscape train and evaluation list you can find them here: [train](https://drive.google.com/file/d/1K9TILq7zmecvuiDNGeHnYaulXhcJaX2N/view?usp=sharing), [val](https://drive.google.com/file/d/1lTOjhGxcAsKS1HtAU-vE3eaTmqwoHF36/view?usp=sharing)

## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet50:** Download the model (the one for BGR) from [link](https://github.com/ruotianluo/pytorch-resnet)

Download them and write the path in **__C.VGG_PATH** and **__C.RESNET50_PATH** at ```lib/model/utils/config.py```.

## Train Example

**Don't forget to install sacred**, you cannot run our code without it.
To run the first step of **Cityscape** -> **FoggyCityscape** on VGG16 with our hyper-parameters :

```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python experiments/exp_traineval_htcn.py
```
If you want to do your own hyper-parameters, etc..., **exp_traineval_htcn.py** is fairly easy to understand to modify. We provided an example to modify this at the end.
To train DTM for the next incremental step, you will have to change 'load_name' inside **exp_train_mask.py** to the model obtained from previous step:
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python experiments/exp_train_mask.py
```
The next incremental domain adaptation, you will have to point 'mask_load_p' to the DTM model you just trained and 'load_name' the model obtained from the first domain adaptation inside **exp_traineval_htcn_inc_mask.py**:
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python experiments/exp_traineval_htcn_inc_mask.py
```

Example of how to edit hyper-parameters etc... in **exp_traineval_htcn.py**:
Original:
```
if __name__ == "__main__":
    ex.run(config_updates={'cfg_file': 'cfgs/vgg16.yml',
                           'lr': 0.001,
                           'lr_decay_step': [5],
                           'max_epochs': 7,
                           'net': 'vgg16',
                           'pretrained': True,
                           'dataset_source': 'cs',
                           'dataset_target': 'cs_fg',
                           'val_datasets': ['cs_fg']},

           options={"--name": 'htcn_cs_2_cs_fg_vgg16'})
```
Edit for **PascalVOC** -> **Clipart** in **exp_traineval_htcn.py**:
```
if __name__ == "__main__":
    ex.run(config_updates={'cfg_file': 'cfgs/resnet50.yml',
                           'lr': 0.001,
                           'lr_decay_step': [5],
                           'max_epochs': 7,
                           'net': 'res50',
                           'pretrained': True,
                           'dataset_source': 'voc_0712',
                           'dataset_target': 'clipart',
                           'val_datasets': ['clipart']},

           options={"--name": 'NAME_OF_YOUR_EXPERIMENT'})
```

## Test

Similar to training, modify **exp_eval.py** to evaluate a model.
Edit 'val_datasets' for the dataset you want to test, it can be multiple datasets, 'model_pth' the path to the model:
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
       python experiments/exp_eval.py 
```

### Trained Model

- **Model** of **Pascal** -> **Clipart** -> **Watercolor**: [Link](https://drive.google.com/file/d/1nDp1bEPaDB-I5nBBjGc3TfPd5etKxVaJ/view?usp=sharing)
- **Model** of **Pascal** -> **Clipart** -> **Watercolor** -> **Comic**: [Link](https://drive.google.com/file/d/1R4FwIoD-mOJ8SS_awZfBgDlY6AZvN3ox/view?usp=sharing)
