
ROAD🛤  
===========
### Enhancing Disease-Free Survival Prediction in Breast Cancer through Synergistic Integration of Multi-Modal Pathological Images and Genomic Data in a Deep Learning Framework
**

*Our study used a combination of images, genes, and clinical data to create a smart computer model that predicts the chances of breast cancer patients staying disease-free. This model works really well, both in training and testing, and can predict accurately over different time periods. This new method could help find cancer recurrence early and customize treatments for patients. Our study is unique because it combines different types of data and uses advanced technology to improve prediction accuracy.*

*We made a computer program that's really good at predicting if breast cancer patients will stay disease-free. It looks at pictures, genes, and patient information. The program did a great job in tests and could be a big help in finding cancer early and giving personalized treatment. Our study is special because it brings together different kinds of data to make better predictions.*

© This code is made available for non-commercial academic purposes. 

<img src="Figure 1_00.png" width="1000px" align="center" />

## Pre-requisites:

* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce V100 x 4)
* Python (3.7.7), h5py (2.10.0), matplotlib (3.1.1), numpy (1.18.1), opencv-python (4.1.1), openslide-python (1.1.1), openslide (3.4.1), pandas (1.0.3), pillow (7.0.0), PyTorch (1.5.1), scikit-learn (0.22.1), scipy (1.3.1), tensorflow (1.14.0), tensorboardx (1.9), torchvision (0.6).

### Installation Guide for Linux (using anaconda)
[Installation Guide](https://github.com/mahmoodlab/CLAM/blob/master/docs/INSTALLATION.md)

### Data Preparation
[Data Preparation](https://github.com/mahmoodlab/TOAD)

### Training
``` shell
CUDA_VISIBLE_DEVICES=0 python main_mtl_concat_dfs.py --drop_out --early_stopping --lr 2e-4 --k 1 --exp_code DFS  --task DFS  --log_data  --data_root_dir DATA_ROOT_DIR --gene True
```
The GPU to use for training can be specified using CUDA_VISIBLE_DEVICES, in the example command, GPU 0 is used. Other arguments such as --drop_out, --early_stopping, --lr, --reg, and --max_epochs can be specified to customize your experiments. 

For information on each argument, see:
``` shell
python main_mtl_concat_dfs.py -h
```

By default results will be saved to **results/exp_code** corresponding to the exp_code input argument from the user. If tensorboard logging is enabled (with the arugment toggle --log_data), the user can go into the results folder for the particular experiment, run:
``` shell
tensorboard --logdir=.
```
This should open a browser window and show the logged training/validation statistics in real time. 

### Evaluation 
User also has the option of using the evluation script to test the performances of trained models. Examples corresponding to the models trained above are provided below:
``` shell
CUDA_VISIBLE_DEVICES=0 python eval_mtl_concat.py --drop_out --k 1 --models_exp_code dummy_mtl_sex_s1 --save_exp_code dummy_mtl_sex_s1_eval --task study_v2_mtl_sex  --results_dir results --data_root_dir DATA_ROOT_DIR
```

For information on each commandline argument, see:
``` shell
python eval_mtl_concat.py -h
```

To test trained models on your own custom datasets, you can add them into **eval_mtl_concat.py**, the same way as you do for **main_mtl_concat_dfs.py**.


## Issues
- Please report all issues on the public forum.

## License
© This code is made available under the GPLv3 License and is available for non-commercial academic purposes. 
