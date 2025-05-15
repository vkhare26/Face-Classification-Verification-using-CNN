# Face-Classification-Verification-using-CNN

# Overview
This project focuses on building a Convolutional Neural Network (CNN) for face classification and verification tasks. Using a ResNet-18 architecture, the model extracts facial embeddings from the VGGFace2 dataset (8631 identities) and verifies identities on a separate dataset (6000 image pairs), leveraging advanced techniques like data augmentation, CutMix, and cosine annealing scheduling.

# Objectives

Face Classification: Train a CNN to classify face images into one of 8631 identities using cross-entropy loss.
Face Verification: Use embeddings to compute similarity scores between image pairs, evaluated via Equal Error Rate (EER) on Kaggle.
Explore CNN architectures, data augmentation, and optimization strategies to improve feature extraction and generalization.

# Methodology

## Dataset:
Classification: Subset of VGGFace2 dataset (112x112 resolution, 8631 identities).
Verification: 6000 image pairs with 5749 identities for validation/testing.


## Preprocessing:
Applied data augmentation (random flips, rotations, color jitter, affine transforms, RandomErasing).
Used CutMix to enhance training robustness.
Normalized images with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5].


## Model:
Implemented ResNet-18 with residual blocks for feature extraction.
Output embeddings (512-dim) for verification; final layer for classification (8631 classes).


## Training:
Optimized with SGD (lr=0.01, momentum=0.9, weight decay=5e-4).
Used CosineAnnealingWarmRestarts scheduler (T_0=10, T_mult=2, eta_min=1e-5).
Trained for 30 epochs with mixed precision (AMP) on GPU.


## Verification:
Computed cosine similarity between embeddings of image pairs.
Evaluated using EER, AUC, and TPR@FPR metrics.



# Results

## Classification: Achieved high accuracy on the dev set (exact numbers depend on training run; best model saved as best_cls.pth).
Verification: Competitive EER on the Kaggle competition (see best_ret_submission.csv for scores).
The model generalizes well to open-set identities, demonstrating robust feature extraction.
