import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import boxcox, normaltest
from math import floor

# Load the dataset
df = pd.read_csv("Datasets/daily_and_month_call_report.csv")

# Display basic dataset information
print("Initial Data Preview:")
print(df.head().to_string())
print(f"Dataset Shape: {df.shape}")

# Handling Missing Values
print("\nMissing Values Before Handling:")
print(df.isnull().sum())

for column in df.columns:
    if df[column].dtype == 'object':  # Categorical columns
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:  # Numerical columns
        df[column].fillna(df[column].mean(), inplace=True)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# Convert Categorical Features to Numeric
categorical_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for category in categorical_columns:
    df[category] = le.fit_transform(df[category])

# Display updated dataset
print( "\nUpdated Data Preview:")
print(df.head().to_string())

# Correlation Analysis
plt.figure(dpi=100)
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Selection for Regression
cdf = df[["Total Number of Calls", "Total Number of Doctors Consultancy"]]

# Splitting Data into Training and Testing Sets
msk_split = np.random.rand(len(df)) <= 0.8
train_df = cdf[msk_split]
test_df = cdf[~msk_split]
print(f"\nTrain Set Shape: {train_df.shape}, Test Set Shape: {test_df.shape}")

# Training the Regression Model
regression = linear_model.LinearRegression()
train_x = np.asarray(train_df[["Total Number of Calls"]])
train_y = np.asarray(train_df[["Total Number of Doctors Consultancy"]])
regression.fit(train_x, train_y)

print("\nLinear Regression Model:")
print(f"Coefficient: {regression.coef_[0][0]:.2f}")
print(f"Intercept: {regression.intercept_[0]:.2f}")

# Model Prediction Example
user_input = float(input("Enter Total Number of Calls: "))
predicted_y = regression.intercept_[0] + regression.coef_[0][0] * user_input
print(f"Predicted Total Number of Doctor Consultations: {floor(predicted_y)}")

# Scatter Plot of Regression Model
plt.scatter(train_df["Total Number of Calls"], train_df["Total Number of Doctors Consultancy"], c="blue")
plt.plot(train_x, regression.coef_[0][0] * train_x + regression.intercept_[0], color="green")
plt.title("Regression Model Visualization")
plt.xlabel("Total Number of Calls")
plt.ylabel("Total Number of Doctor Consultations")
plt.show()

# Model Testing & Accuracy
test_x = np.asarray(test_df[["Total Number of Calls"]])
test_y = np.asarray(test_df[["Total Number of Doctors Consultancy"]])
test_prediction = regression.predict(test_x)
print(f"\nR² Score: {r2_score(test_y, test_prediction):.2f}")

# Normalization & D'Agostino K² Test
def apply_transformations(data, column):
    transformations = {
        "Original": data[column],
        "Log": np.log1p(data[column]),
        "Square Root": np.sqrt(data[column]),
        "Box-Cox": boxcox(data[column] + 1)[0]
    }
    
    results = {}
    for key, transformed in transformations.items():
        p_value = normaltest(transformed)[1]
        results[key] = p_value
    
    return pd.DataFrame.from_dict(results, orient='index', columns=['p-value'])

# Apply transformations and display results
normalization_results = apply_transformations(df, "Total Number of Calls")
print("\nNormalization & D'Agostino K² Test Results:")
print(normalization_results)
