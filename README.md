# Airbnb Price Prediction: Combining CNN, RNN, Regression, and Regularization

![Python](https://img.shields.io/badge/Python-3.8-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Introduction
This project aims to predict Airbnb rental prices by combining multiple data modalities:
- **Structured data**: Features like location, amenities, and property type.
- **Images**: Visual analysis of property photos using Convolutional Neural Networks (CNN).
- **Text**: Contextual understanding of user reviews using Recurrent Neural Networks (RNN).
- **Regularization**: Techniques like Dropout, L2 Regularization, and Early Stopping to prevent overfitting.

The goal is to demonstrate how to integrate diverse data types into a unified predictive model.

## Dataset
The dataset used in this project is sourced from [Kaggle's Airbnb Price Prediction](https://www.kaggle.com/datasets). It includes:
- **Structured data**: CSV files with features like price, location, and amenities.
- **Images**: Photos of the properties.
- **Text**: User reviews and comments.

## Approach
The project combines the following techniques:
1. **CNN**: Extracts visual features from property images.
2. **RNN (LSTM)**: Processes sequential text data from user reviews.
3. **Regression**: Handles structured data to predict prices.
4. **Regularization**: Improves model robustness and generalization.

Here’s a high-level overview of the pipeline:
