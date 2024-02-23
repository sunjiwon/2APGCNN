# 2APGCNN
Advanced Protein Aggregation Prediction Graph Convolutional Neural Network (2APGCNN)

## Overview

This repository contains the implementation of a Graph Convolutional Network (GCN) for predicting protein aggregation (PA) scores, a critical phenomenon associated with neurodegenerative diseases such as Alzheimer's and Parkinson's. The model is trained on an expanded and refined dataset obtained from the Protein Data Bank (PDB) and AlphaFold2.0.

## Dataset Preparation

To enhance the dataset, we utilized AGGRESCAN3D 2.0 to calculate PA propensity. Multi polypeptide chains within PDB data were systematically separated into single polypeptide chains, resulting in a dataset comprising 302,032 unique PDB entries. Additionally, 22,774 Homo sapiens data from AlphaFold2.0 were included.

## Model Performance

The trained GCN model achieved an impressive coefficient of determination (R2) score of 0.99 and low mean absolute error (MAE). This demonstrates the effectiveness of incorporating structural information into the model for accurate PA prediction.

## Active Learning

We implemented an active learning process to rapidly identify proteins with high PA propensity. The active learning approach outperformed other methods, achieving a MAE of 0.0291 in expected improvement. It identified 99% of the target proteins by exploring only 29% of the entire search space.

## Usage

- Clone the repository: `git clone [repository_url]`
- Install dependencies: `pip install -r requirements.txt`
- Run the prediction script: `python predict.py`

## Contribution

We welcome contributions and bug reports. If you find any issues or have suggestions, please open an issue or create a pull request.

## Citation

If you use this work in your research, please cite our paper [Link to your paper].

## License

This project is licensed under the [License Name] - see the [LICENSE.md](LICENSE.md) file for details.
