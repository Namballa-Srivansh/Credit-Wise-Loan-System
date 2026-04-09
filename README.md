# Credit-Wise Loan System

This repository contains a machine learning project that predicts whether a loan should be approved or not based on various applicant features. The model is built using Python and the `scikit-learn` library.

## Project Overview

The project uses a dataset (`loan approval data.csv`) to train a **Gaussian Naive Bayes** classifier. The pipeline includes:
- **Data Preprocessing:** Handling missing values using mean imputation for numerical columns and most-frequent imputation for categorical columns.
- **Categorical Encoding:** Using `LabelEncoder` for ordinal features and `OneHotEncoder` for nominal features.
- **Feature Scaling:** Standardizing the features using `StandardScaler`.
- **Exploratory Data Analysis (EDA):** Scripts for visualizing data distribution and feature correlation to understand the importance of different features (available in the code as commented-out sections).
- **Model Training & Evaluation:** Training the Naive Bayes model and evaluating it using precision, recall, F1 score, accuracy, and a confusion matrix.

## Dataset

The dataset used is `loan approval data.csv`. It contains information such as Applicant Income, Credit Score, DTI Ratio, Savings, Education Level, Employment Status, Marital Status, Loan Purpose, Property Area, Gender, and Employer Category.

## Requirements

To run this project, make sure you have the following Python libraries installed:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install the required packages using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## How to Run

1. Clone the repository and navigate into the project directory.
2. Ensure you have the `loan approval data.csv` file in the same directory.
3. Run the Python script:

```bash
python model.py
```

*Note: Some evaluation and plotting codes are commented out in `model.py`. You can uncomment those blocks to see the exploratory data analysis plots and model performance metrics in your console/GUI.*
