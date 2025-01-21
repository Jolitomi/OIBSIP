# Retail Sales Data Exploratory Data Analysis (EDA)

## Introduction
This project involves an exploratory data analysis (EDA) of retail sales data to uncover trends, patterns, and actionable insights. The dataset includes information on transactions, customer demographics, product categories, and sales amounts, which are analyzed to provide strategic recommendation
---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Preprocessing Steps](#preprocessing-steps)
3. [Data Visualization](#data-visualization)
4. [Key Insights](#key-insights)
5. [Recommendations](#recommendations)
6. [How to Use]-to-use)

---

## Dataset Overview
The dataset contains the following columns:
- **Transaction ID**: Unique identifier for each transaction.
- **Date**: The date of the transaction.
- **Customer ID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Product Category**: Category of the product purchased.
- **Quantity**: Number of units purchased.
- **Price per Unit**: Price of a single unit.
- **Total Amount**: Total amount spent (Qty × Price per Unit).

---

## Preprocessing Steps
### 1. Importing Necessary Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```
**Explanation**: Import libraries for data manipulation, visualization, and statistical analysis.

### 2. Loading the Dataset
```python
df = pd.read_csv("retail_sales_dataset.csv")
print("Dataset Info:")
print(df.info())
```
**Outcome**: No missing values are detected, and the `Date` column is converted to `datetime` for time-based analysis.

### 3. Data Cleaning
- Checked for duplicates: `df.duplicated().sum()` (Result: 0).
- Validated `Total Amount` calculation: Ensured `Total Amount = Quantity × Price per Unit`.

---

## Data Visualization
### Gender Distribution
```python
gender_counts = df['Gender'].value_counts()
gender_counts.plot(kind='bar', color=['pink', 'skyblue'], title='Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
```
**Insight**: Females made slightly more purchases than males.

### Sales by Product Category
```python
category_sales = df.groupby('Product Category')['Total Amount'].sum()
category_sales.plot(kind='bar', color='orange', title='Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.show()
```
**Insight**: `Clothing` and `Electronics` are the highest revenue-generating categories.

### Age Distribution
```python
bins = np.arange(0, 101, 10)
age_groups = pd.cut(df['Age'], bins=bins, right=False)
age_counts = age_groups.value_counts().sort_index()
age_counts.plot(kind='bar', color='green', title='Age Distribution')
.xlabel('Age Groups')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
```
**Insight**: The majority of customers fall into the 30–50 age range.

### Sales Trend Over Time
```python
sales_trend = df.groupby('Date')['Total Amount'].sum()
sales_trend.plot(kind='line', title='Sales Trend Over Time', color='blue')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plow()
```
**Insight**: Peaks in May and December suggest sales promotions or holiday effects.

---

## Key Insights
1. **High-Revenue Product Categories**:
   - `Clothing` and `Electronics` dominate revenue.
   - `Electronics` shows higher margins per unit, while `Clothing` sells in larger quantities.

2. **Gender-Specific Purchasing Patterns**:
   - Males prefer `Electronics`.
   - Females dominate `Clothing` and `Beauty`.

3. **Seasonal Trends**:
   - Sales peak during holi and promotions.

4. **Age Distribution**:
   - Customers aged 30–50 contribute the most consistent revenue.

---

## Recommendations
1. **Boost Electronics and Clothing Sales**:
   - Offer product bundles for `Electronics`.
   - Use data on popular sizes for targeted inventory planning in `Clothing`.

2. **Gender-Specific Promotions**:
   - Promote gadgets to males during tech sales events.
   - Bundle `Clothing` and `Beauty` products for females.

3. **Seasonal Campaigns**:
   - Focus on holiday promotions (e.g., Black Friday).
   - Offer loyalty discounts before peak seasons.

4. **Engage HSpending Age Groups**:
   - Create milestone-based rewards for 30–50-year-olds.
   jolitomiuOIBSIP/dent dscouns for young adults.

---

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/retail-sales-eda.git
   ```
2. Navigate to the project directory:
   ```bash
   cd retail-sales-eda
   ```
3. Install necessary libraries:
   ```bash
   pip install -r rrements.txt
   ```
4. Run the analysis:
   ```bash
   python analysis_script.py
   ```
5. View results and visualizations in the `output` folder.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.




```python

```
