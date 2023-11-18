# mask-Tn-S1-Network
Vision Transformer Model of Multi teacher Knowledge Distillation and Self supervised Learning.
## Training
### Documentation
Please install PyTorch and download cifar-10 dataset, Fashion MNIST dataset, and ImageNet dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. 
### Training of DINO model
Here we are using the source code of the DINO model trained on three datasets.
```
python main_dino.py
```
### Training of Mask-T2-S1 model
The visual converter model for self supervised learning of two teacher networks and one student network based on mask mechanism and multi teacher knowledge distillation used in our article was trained on three datasets.
```
python main_dino_t2 Network.py
```
### Training of Mask-T3-S1 model
The visual converter model for self supervised learning of three teacher networks and one student network based on mask mechanism and multi teacher knowledge distillation used in our article was trained on three datasets.
```
python main_dino_t3 Network.py
```
## Evaluation
This article adopts the method of directly adding output tensors while keeping the feature dimensions unchanged to evaluate the architecture based on mask mechanism and multi teacher knowledge distillation
### Evaluation of the DINO model
```
python linear.py
```
### Evaluation of the Mask-T2-S1 model
Evaluate a visual converter model for self supervised learning of two teacher networks and one student network based on mask mechanism and multi teacher knowledge distillation on multiple datasets using the method of directly adding feature dimensions.
```
python linear-t2.py
```
### Evaluation of the T3-S1 model
Evaluate a visual converter model for self supervised learning of three teacher networks and one student network based on mask mechanism and multi teacher knowledge distillation on multiple datasets using the method of directly adding feature dimensions.
```
python linear-t3.py
```


