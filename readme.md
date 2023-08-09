### MCA-GCN

MCA-GCN:An Explainable Method for Alzheimer's Disease Diagnosis with Brain Imaging Genetic Data by Incorporating Multi-stream Attention Mechanisms and Graph Convolutional Networks

#### Requirements

- pytorch==1.12.1+cu113
- numpy==1.23.5
- scipy==1.10.1
- scikit-learn==1.2.2
- tqdm==4.65.0
- positional_encodings==6.0.1 
  
  In addition, CUDA 11.3 have been used on NVIDIA GeForce RTX 3090.

#### Data

Due to privacy and ethical considerations and ADNI's data usage agreement, users need to apply at ADNI and download these data(https://adni.loni.usc.edu/)

fMRI preprocessing:

> DPARSF(http://rfmri.org/DPARSF)

SNP preprocessing:

> 1.PLINK(https://zzz.bwh.harvard.edu/plink/index.shtml)               2.MAGMA(https://ctg.cncr.nl/software/magma)