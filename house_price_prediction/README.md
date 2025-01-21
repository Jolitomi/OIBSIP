# House Price Prediction Model

## **Objective**
The objective of this project is to build a predictive model using linear regression to estimate house prices based on a dataset of various housing features. The model uses a set of factors like area, number of bedrooms, bathrooms, and other relevant features to predict the price of a house.

---

## **Table of Contents**
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
   - [1. Load the Dataset](#1-load-the-dataset)
   - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   - [3. Data Preprocessing](#3-data-preprocessing)
   - [4. Train-Test Split](#4-train-test-split)
   - [5. Train the Model](#5-train-the-model)
   - [6. Evaluate the Model](#6-evaluate-the-model)
   - [7. Visualize Results](#7-visualize-results)
   - [8. Save the Model (Optional)](#8-save-the-model-optional)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [Contributing](#contributing)
7. [License](#license)

---

## **Installation**

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/jolitomi/OIBSIP/main/house_price_prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

---

## **Dataset**

The dataset `Housing.csv` used in this project contains information about various housing features that can be used to predict house prices. It includes the following columns:

- `area`: The area of the house (in square feet).
- `bedrooms`: The number of bedrooms.
- `bathrooms`: The number of bathrooms.
- `stories`: The number of floors/stories in the house.
- `mainroad`: Whether the house is on the main road (Yes/No).
- `guestroom`: Whether the house has a guestroom (Yes/No).
- `basement`: Whether the house has a basement (Yes/No).
- `hotwaterheating`: Whether the house has hot water heating (Yes/No).
- `airconditioning`: Whether the house has air conditioning (Yes/No).
- `parking`: The number of parking spaces available.
- `prefarea`: Whether the house is in a preferred area (Yes/No).
- `furnishingstatus`: The status of the house’s furnishings (furnished, semi-furnished, unfurnished).
- `price`: The target variable representing the price of the house.

---

## **Workflow**

### **1. Load the Dataset**
```python
import pandas as pd

df = pd.read_csv('Housing.csv')
print(df.head())
```

### **2. Exploratory Data Analysis (EDA)**

This step helps to understand the dataset's structure, distribution of features, and relationships between features and the target variable.

```python
# Display dataset information
print(df.info())

# Summary statistics
print(df.describe())
```

- Visualize the target variable distribution (`price`) and the feature distributions.
- Perform correlation analysis to identify relationships with the target variable.
- Analyze categorical features and their relationship with the target variable.
  
### **3. Data Preprocessing**

This step involves cleaning the data, handling missing values, encoding categorical variables, and preparing the features for model training.

```python
# Handle missing values
print(df.isnull().sum())

# Encode binary categorical variables
df['mainroad'] = df['mainroad'].apply(lambda x: 1 if x == 'yes' else 0)

# One-hot encoding for multi-category variables
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
```

### **4. Train-Test Split**
Split the dataset into training and testing sets (80% train, 20% test).

```python
from sklearn.model_selection import train_test_split

X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **5. Train the Model**
We use linear regression to fit the training data and learn the relationship between features and the target variable.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

### **6. Evaluate the Model**

Make predictions using the trained model and calculate performance metrics like Mean Squared Error (MSE) and R-squared (R²).

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
```

### **7. Visualize Results**
Visualize the predicted values against actual values and analyze residuals.

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
```

### **8. Save the Model (Optional)**

You can save the trained model for future use.

```python
import joblib

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')

# Load the model
loaded_model = joblib.load('linear_regression_model.pkl')
```

---

## **Usage**

After training the model, you can make predictions on new input data. Example:

```python
test_input = {
    'area': 3000,
    'bedrooms': 4,
    'bathrooms': 3,
    'stories': 2,
    'mainroad': 1,
    'guestroom': 0,
    'basement': 1,
    'hotwaterheating': 0,
    'airconditioning': 1,
    'parking': 2,
    'prefarea': 1,
    'furnishingstatus_semi-furnished': 1,
    'furnishingstatus_unfurnished': 0
}

test_df = pd.DataFrame([test_input])
predicted_price = loaded_model.predict(test_df)
print(f"Predicted Price: {predicted_price[0]:,.2f}")
```

---

## **Dependencies**
- `pandas` 
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `numpy`

---

## **Contributing**

Feel free to fork this project, create a branch, and submit a pull request with your improvements.

---

## **License**

This project is licensed under the MIT License.

