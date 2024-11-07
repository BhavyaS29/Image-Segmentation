# Image-Segmentation
Problem Statement : Image Segmentation in Surround View Images
# Overview
This repository contains code for performing image segmentation on surround view images using a U-Net model. The project leverages the Cityscapes Image Pairs dataset from Kaggle and includes fine-tuning on a custom dataset with reduced classes, focusing on segmenting cars and the background. The main objective is to accurately segment objects in images captured from multiple surrounding cameras, which is crucial for applications in autonomous driving and surveillance.
# Project Description
The project involves using convolutional neural networks (CNNs), specifically a U-Net model, for segmenting objects in the images. The Kaggle notebooks contain the full image segmentation code, including data preprocessing, model training, and evaluation scripts and also the code to reduce the number of classes in the custom dataset to two. Both notebooks contain step-by-step instructions and explanations for each part of the process.
# Hardware Requirements
A computer with at least 8GB of RAM (16GB recommended for large datasets)  
GPU (NVIDIA GPU with CUDA support is highly recommended for faster training)
# Software Requirements
Python 3.7 or higher  
Jupyter Notebook or Jupyter Lab  
Kaggle API (for downloading datasets from Kaggle)  
TensorFlow or PyTorch (depending on which framework you used)  
OpenCV  
NumPy  
Matplotlib  
Scikit-learn  
# Usage
You can run the notebooks directly in Kaggle or Colab. For local execution:  
1.Launch Jupyter Notebook or Jupyter Lab:  
  jupyter notebook  
2.Open the desired notebook and follow the instructions provided within the notebook.  
3.Ensure that your environment is configured to use a GPU for optimal performance if available.  
