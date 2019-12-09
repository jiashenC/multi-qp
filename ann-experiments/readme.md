# Approximate Nearest Neighbour Experiments

This folder consists of the experiments run for improving the performance during inference using approximate nearest neighbours instead of exact nearest neighbours.


# Requirements

[PyTorch](https://pytorch.org/get-started/locally/)
[faiss](https://github.com/facebookresearch/faiss)
[annoy](https://github.com/spotify/annoy)

# NVIDIA AI City Challenge Dataset

Request for the dataset access at the official [NVIDIA AI City Challenge](https://www.aicitychallenge.org/) for track 2.

Unzip the Dataset in the folder `ann-experiments`

# Running the Experiments

Run `train_triplet_network.py` to create the baseline model for evaluating ANN performance.

Run `evaluate_full_pairwise_compute.py` to evaluate baseline exact nearest neighbour performance.

Run `evaluate_faiss.py` to evaluate the performance of faiss library.

Run `evaluate_annoy.py` to evaluate the performance of annoy library.
