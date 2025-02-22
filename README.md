# Predictive Maintenance Using Machine Learning & Deep Learning

## Overview
This project focuses on **predictive maintenance** using **machine learning and deep learning models**. The goal is to anticipate **vehicle maintenance needs** based on historical data, helping fleet management optimize **resource allocation, reduce downtime, and improve operational efficiency**.

## Key Features
- **Binary Classification:** Predict whether a unit will require maintenance (`Yes/No`).
- **Time-Series Forecasting:** Estimate the number of maintenance events in a given window.
- **Feature Engineering:** Transform raw data into valuable predictors.
- **Data Augmentation:** Use **SMOTE** to address class imbalance.
- **LSTM Implementation:** Train a recurrent neural network for sequential forecasting.

##  Libraries Used
This project utilizes several **data science, machine learning, and deep learning** libraries:

```python
import pandas as pd  # Data manipulation
import numpy as np  # Numerical computations
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Statistical visualizations

from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier  # Tree-based models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay  # Model evaluation
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Model selection & hyperparameter tuning
from sklearn.decomposition import PCA  # Dimensionality reduction
from sklearn.preprocessing import MinMaxScaler  # Feature scaling
from imblearn.over_sampling import SMOTE  # Handling class imbalance
from xgboost import XGBClassifier  # Gradient boosting model

import tensorflow as tf  # Deep learning framework
from tensorflow.keras.models import Sequential  # Model architecture
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, LSTM  # Neural network layers
from tensorflow.keras.regularizers import l2  # Regularization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Learning rate adjustments
from tensorflow.keras.optimizers import Adam  # Optimizer
