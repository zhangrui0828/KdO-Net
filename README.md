# KdO-Net

This repository represents the official implementation of  KdO-Net.

## Instructions

### Requirements

The code is based on Python3  and is implemented in Tensorflow. The required libraries can be easily installed by runing

```shell
pip install -r requirements.txt
```

in your environment.

### Training

To train the network with the following code：
```shell
cd KdO-Net
python train.py
```

The training data should be saved in folder `data/train/trainingdata`.The training generated model file is saved in `./models/` and the tensorboard log will be saved in `./KdO-Net/logs/`.For more training options please see `./KdO-Net/core/config.py`.

### Evaluation

The source-code for the performance evaluation on the 3DMatch data set is available in the `./KdO-Net/Test/evaluate.py`.Evaluate the model e.g., using

```shell
cd KdO-Net
python test.py
cd Test
python evaluate.py
```

`test.py` is used to infer the feature descriptors, run the `evaluate.py`to compute the recall ,precision et. al.

Before running `test.py`, the preprocessed test set data needs to be saved in  folder `./KdO-Net/data/3DMatch/test`. For more options in runing the inference please see `./KdO-Net/core/config.py`.

### Demo

To carry out the demo, please run

```shell
cd KdO-Net
python demo.py
```

### About

If you have any questions, please feel free to discuss in the issues.
