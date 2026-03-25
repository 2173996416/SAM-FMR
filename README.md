# SAM-FMR
Label-Efficient Mapping of Unregulated Waste Dumps via Mixed Supervision with Feature Masked Recovery

# Introduction
This repository provides the implementation of a weakly supervised semantic segmentation framework for unregulated waste dumps in high-resolution remote sensing images. The method is built upon the Conformer backbone and incorporates SAM (Slice-and-Merge) and FMR (Feature Masked Recovery) to improve the completeness and quality of class activation maps (CAMs), thereby enhancing the final segmentation performance.

The repository includes scripts for model training, testing, CAM generation, CAM evaluation, and segmentation evaluation.

# Datasets
The experiments are conducted on two datasets:                                                                          
1.YREB Dataset                                                                                                          
2.Global Dumpsite

# Main Files
train.py: Main script for model training.                                                                               
test.py: Script for model inference and testing.                                                                        
conformer_sam_fmr.py: Implementation of the proposed model with SAM and FMR.                                            
FMR_module.py: Definition of the Feature Masked Recovery module.                                                        
cam_evaluation.py: Quantitative evaluation for generated CAMs.                                                          
seg_eval.py: Quantitative evaluation for segmentation results.

# Usage
# Training
Run the training script:                                                                                                
python train.py
# Testing
python test.py
# CAM Evaluation
python cam_evaluation.py
# Segmentation Evaluation
python seg_eval.py
