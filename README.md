# Airbnb Price Prediction: Combining CNN, RNN, Regression, and Regularization

![Python](https://img.shields.io/badge/Python-3.8-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview
This project predicts Airbnb rental prices using a multimodal approach that combines:
- **Structured data**: Features like number of bedrooms, bathrooms, and reviews.
- **Images**: Thumbnails of property photos extracted from URLs in the dataset.
- **Textual features**: Descriptions of properties written by hosts.

The pipeline integrates advanced machine learning techniques:
- **Regression** for structured data.
- **Convolutional Neural Networks (CNN)** for image analysis.
- **Recurrent Neural Networks (RNN)** for processing textual descriptions.
- **Regularization techniques** (Dropout, L2 Regularization, Early Stopping) to improve model robustness.

---

## 📊 Dataset
The dataset used in this project is sourced from [Kaggle's Airbnb Data](https://www.kaggle.com/datasets). It includes:
- **Structured data**: CSV file with features like `log_price`, `accommodates`, `bedrooms`, `review_scores_rating`, etc.
- **Image URLs**: Thumbnails of property photos (`thumbnail_url`).
- **Textual data**: Property descriptions (`description`) and amenities (`amenities`).

### Key Columns
| Column Name            | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `log_price`            | Logarithmic price of the property (target variable).                        |
| `property_type`        | Type of property (e.g., apartment, house).                                  |
| `room_type`            | Type of room (e.g., entire home, private room).                             |
| `accommodates`         | Number of people the property can accommodate.                              |
| `bathrooms`            | Number of bathrooms.                                                       |
| `bedrooms`             | Number of bedrooms.                                                        |
| `description`          | Textual description of the property written by the host.                   |
| `thumbnail_url`        | URL of the property's thumbnail image.                                     |
| `amenities`            | List of amenities provided by the property (e.g., pool, Wi-Fi).            |

---

## 🔧 Approach
The project follows these steps:

1. **Data Preparation**:
   - Clean and preprocess structured data (normalization, handling missing values).
   - Download and preprocess images from `thumbnail_url`.
   - Tokenize and encode textual descriptions (`description`).

2. **Modeling**:
   - Use a regression model for structured data.
   - Apply a CNN to extract visual features from images.
   - Use an RNN (LSTM/GRU) to process textual descriptions.
   - Combine outputs from all models into a final prediction layer.

3. **Regularization**:
   - Add Dropout layers to prevent overfitting.
   - Use L2 Regularization on dense layers.
   - Implement Early Stopping during training.

4. **Evaluation**:
   - Evaluate the model using metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

---

## 🚀 Installation
   -Clone the repository:
   ```bash
   git clone https://github.com/hafs96/Airbnb-Price-Prediction-Combining-CNN-RNN-Regression-and-Regularization.git
   cd airbnb-price-prediction
