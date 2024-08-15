# 2APGCNN: Advanced Protein Aggregation Prediction Graph Convolutional Neural Network (2APGCNN)

## Overview

This repository provides the implementation of a Graph Convolutional Network (GCN) designed to predict protein aggregation (PA) scores. Protein aggregation is a key factor in neurodegenerative diseases such as Alzheimer's and Parkinson's. Our model leverages an enriched dataset sourced from the Protein Data Bank (PDB) and AlphaFold2.0 to achieve high predictive accuracy.

## Requirement

We recommend using Python 3.7 or higher. Install the required libraries using the commands below:

####pip install torch==1.11.0+cu111 torchvision==0.12.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
####pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
####pip install torch-geometric
####conda install -y -c rdkit rdkit
####pip install biopython
####conda install -y -c conda-forge biotite
####pip install tqdm

###Note: Your environment might differ, so adjust the versions and install any additional libraries as required.


## Dataset Preparation

We enhanced the dataset using AGGRESCAN3D 2.0 to calculate PA propensity. Multi-polypeptide chains from PDB data were separated into individual chains, resulting in 302,032 unique entries. Additionally, data from 22,774 Homo sapiens proteins were included from AlphaFold2.0.

## Model Performance

Our trained GCN model achieved an outstanding R2 score of 0.99 and a low Mean Absolute Error (MAE), highlighting the effectiveness of incorporating structural data for accurate PA prediction.

## Active Learning

An active learning strategy was implemented to identify proteins with high PA propensity efficiently. This approach outperformed other methods, achieving a MAE of 0.0291 in expected improvement and identifying 99% of target proteins by exploring only 29% of the search space.

## Citation

If you use this work in your research, please cite our paper [Link to your paper].

