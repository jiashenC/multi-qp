# Knowledge Distillation

This folder consists of the experiments conducted on how knowledge distillation improve inference time performance on Re-Identification tasks.

Read more about it in the [original paper here](https://arxiv.org/abs/1503.02531).

# Requirements

Install `torchreid` by following instructions from [here](https://github.com/KaiyangZhou/deep-person-reid)

[PyTorch](https://pytorch.org/get-started/locally/)

# Running the experiments
To Train a baseline model on market1501 dataset,

`python train_baseline.py --model_type resnet50 --num_epochs 30`

To Train a distilled model on market1501 dataset from a teacher model,

`python train_knowledge_distillation_model.py --model_type squeezenet1_0 --teacher_type resnet50 --num_epochs 30`

Ensure that you have the pretrained teacher model for the given dataset present in the same path for distilled training.