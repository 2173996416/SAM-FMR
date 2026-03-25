# SAM-FMR
Label-Efficient Mapping of Unregulated Waste Dumps via Mixed Supervision with Feature Masked Recovery

## Introduction
This repository provides the implementation of a weakly supervised semantic segmentation framework for unregulated waste dumps in high-resolution remote sensing images. The method is built upon the Conformer backbone and incorporates SAM (Slice-and-Merge) and FMR (Feature Masked Recovery) to improve the completeness and quality of class activation maps (CAMs), thereby enhancing the final segmentation performance.

The repository includes scripts for model training, testing, CAM generation, CAM evaluation, and segmentation evaluation.

## Getting Started
### Requirements
Create a virtual environment and install the necessary dependencies:<br>
`conda create -n conformer-cam python=3.8`<br>
`conda activate conformer-cam`<br>
`pip install -r requirements.txt`

### Datasets
The experiments are conducted on two datasets:<br>
1.YREB Dataset<br>
2.Global Dumpsite

## Usage
### Training                                                                                             
`python train.py`
### Testing
`python test.py`
### CAM Evaluation
`python cam_evaluation.py`
### Segmentation Evaluation
`python seg_eval.py`
