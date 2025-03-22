# Data Preprocessing and Regression Analysis

## Overview
This project focuses on data preprocessing, correlation analysis, regression modeling, and normality transformation techniques. It includes:
- Handling missing values
- Encoding categorical variables
- Performing correlation analysis
- Implementing Simple Linear Regression
- Evaluating model performance
- Applying normalization transformations and D'Agostino K² normality test

## Requirements
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy scikit-learn scipy seaborn matplotlib
```

## Data Preprocessing
1. **Handling Missing Values:**
   - Categorical values are replaced with the most frequent value.
   - Numerical values are replaced with the mean.
2. **Encoding Categorical Variables:**
   - `LabelEncoder` is used to convert categorical values into numeric form.
3. **Correlation Analysis:**
   - A heatmap is plotted to visualize feature relationships.

## Regression Model
1. **Defining Features & Target Variable:**
   - Independent variable: `Total Number of Calls`
   - Dependent variable: `Total Number of Doctors Consultancy`
2. **Splitting the Dataset:**
   - 80% training data, 20% testing data
3. **Training the Model:**
   - Simple Linear Regression using `LinearRegression()`
4. **Making Predictions:**
   - The model predicts the number of doctor consultations based on user input.
5. **Model Evaluation:**
   - `r2_score()` is used to measure performance.
   - Regression line is plotted along with scatter points.

## Normalization and D’Agostino K² Test
1. **Applying Transformations:**
   - Log Transformation
   - Square Root Transformation
   - Box-Cox Transformation (for positive values only)
2. **Performing Normality Test:**
   - `normaltest()` function is used to evaluate whether data follows a normal distribution.
   - Results are displayed in a structured table.

## Potential Enhancements
- Implementing **Feature Scaling** (`MinMaxScaler` or `StandardScaler`).
- Handling **Outliers** using median-based imputation.
- Applying **One-Hot Encoding** for categorical variables.
- Improving regression performance using **alternative error metrics** (`MAE`, `RMSE`).

## Usage
Run the script and follow the prompts to input `Total Number of Calls`. The model will predict the estimated number of doctor consultations.


