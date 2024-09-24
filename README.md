# Student Score Prediction

## Introduction

This project aims to predict student scores in mathematics based on various independent variables. The dataset includes the following features:

- **Gender**
- **Race/Ethnicity**
- **Parental Level of Education**
- **Lunch**
- **Test Preparation Course**
- **Reading Score**
- **Writing Score**

The target variable for this prediction task is the **Math Score**.

## Project Approach

### 1. Data Ingestion

The first step involves reading the dataset from a CSV file. We then split the data into training and test sets.

### 2. Data Transformation

The dataset contains five categorical variables and two numerical variables. We apply the following transformations:

- **Numerical Variables**: For numerical variables, we use `SimpleImputer` with a median strategy to handle missing values, followed by standard scaling to normalize the data.
- **Categorical Variables**: For categorical variables, we again employ `SimpleImputer` using the most frequent strategy. Subsequently, one-hot encoding is applied to convert categorical data into a suitable format for model training.

The preprocessor is saved as a pickle file in the `artifacts` folder.

### 3. Model Training

In this phase, we experiment with various machine learning models and perform hyperparameter tuning to identify the best-performing model. Ultimately, the **Decision Tree** model yields the best results. This model is saved as a pickle file for future use.

### 4. Prediction Pipeline

We create a prediction pipeline where incoming data is transformed into a DataFrame. The pipeline includes functions to load the necessary pickle files and generate predictions based on the input data.

### 5. Deployment using Azure

The trained model is deployed as an Azure Web App, allowing users to access the prediction functionality online.

**Deployment Link**: [will be updated soon]
