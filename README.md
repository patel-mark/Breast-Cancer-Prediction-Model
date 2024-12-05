# Breast Cancer Diagnosis Prediction Model

## Overview

This project implements a machine learning model using PyTorch to predict breast cancer diagnoses based on various medical features. The model uses a neural network classifier to distinguish between malignant and benign breast cancer diagnoses.

## Features

- Data preprocessing and cleaning
- Feature scaling
- Custom PyTorch dataset creation
- Neural network classification
- Model training and validation
- Performance evaluation

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- torch

## Project Structure

The notebook contains several key components:

1. **Data Loading and Preprocessing**
   - Reads data from a CSV file
   - Handles missing values
   - Converts categorical variables
   - One-hot encodes categorical features

2. **Custom PyTorch Dataset**
   - Creates a custom dataset class for breast cancer data
   - Converts features and labels to PyTorch tensors

3. **Neural Network Model**
   - Implements a multi-layer neural network classifier
   - Uses ReLU activation and dropout for regularization
   - Sigmoid output for binary classification

4. **Training Process**
   - Supports GPU acceleration
   - Implements early stopping
   - Tracks training and validation loss
   - Saves the best performing model

## Model Architecture

The neural network consists of:
- Input layer
- First hidden layer: 64 neurons with ReLU activation (30% dropout)
- Second hidden layer: 32 neurons with ReLU activation (20% dropout)
- Output layer: Single neuron with sigmoid activation

## Performance Metrics

In the final evaluation:
- Overall Accuracy: 90%
- Precision for Benign (Class 0): 0.84
- Precision for Malignant (Class 1): 1.00
- Recall for Benign: 1.00
- Recall for Malignant: 0.80

## Usage

1. Ensure all dependencies are installed
2. Run the notebook
3. The script will:
   - Load and preprocess the data
   - Split data into training, validation, and test sets
   - Train the model
   - Evaluate the model's performance
   - Save the best performing model as 'best_model.pth'

## Limitations and Considerations

- Model performance may vary with different datasets
- Small dataset size might limit generalizability
- Requires further validation with medical professionals

## Future Improvements

- Collect more training data
- Experiment with different model architectures
- Implement cross-validation
- Add more feature engineering techniques
