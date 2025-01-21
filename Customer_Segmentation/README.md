# iFood Customer Segmentation: Unlocking Customer Insights  

## Project Overview  
This project dives into the behavior and purchasing patterns of iFood customers. Using advanced data analysis and clustering techniques, we’ve grouped customers into meaningful segments. The goal? To empower data-driven marketing strategies and improve customer targeting.  


## The Dataset  
The dataset (`ifood_df.csv`) consists of **2205 customers** and **39 attributes** covering everything from demographics to spending behavior. Here are some highlights:  
- **Income**: Annual earnings of the customer.  
- **Children at Home**: Number of kids (`Kidhome`) and teens (`Teenhome`) living in the household.  
- **Recency**: Days since the last purchase.  
- **Spending Habits**: Total spend (`MntTotal`) and breakdown by categories like wines, fruits, meats, and more.  
- **Campaign Interaction**: Responses to previous campaigns (e.g., `AcceptedCmp1`, `AcceptedCmp2`).  
- **Purchases**: Total purchases across different channels (`TotalPurchases`).  

Derived features include:  
- **AcceptedCmpOverall**: Total number of campaigns accepted by each customer.  
- **Age_Group**: Customers grouped into relevant age brackets.  


## Step-by-Step Breakdown  

### 1. **Data Preparation**  
The raw data wasn’t perfect, but we made it work:  
- Ensured data consistency—no missing values were found.  
- Addressed outliers (e.g., unusually low or zero incomes).  
- Engineered new features, such as total campaign responses (`AcceptedCmpOverall`) and combined spending totals.  

### 2. **Exploratory Data Analysis (EDA)**  
EDA uncovered valuable insights:  
- **Spending Distribution**: Most customers spend modestly, but a few outliers are high rollers.  
- **Campaign Responses**: Participation varied, with some campaigns seeing more engagement than others.  
- **Customer Demographics**: Explored how age, marital status, and household composition impact spending.  

Key visualizations include:  
- Spending patterns by **Age Group** and **Marital Status**.  
- **Bar plots** for campaign responses.  
- **Heatmaps** to uncover correlations between income, spending, and purchases.  

### 3. **Clustering the Customers**  
The core of this project was clustering customers using **K-Means**:  
- Selected features: `MntTotal`, `Recency`, `TotalPurchases`, and `Income`.  
- Standardized data for better clustering performance.  
- Used the **Elbow Method** to determine the optimal number of clusters.  

**Result**: Two distinct customer segments emerged.  
- **Cluster 0**: The high-value customers—big spenders with high income and frequent purchases.  
- **Cluster 1**: The budget-conscious group—lower spenders with limited purchasing activity.  


## Results at a Glance  

| Segment | Avg Spend (`MntTotal`) | Avg Income | Avg Purchases | Size (%) |  
|---------|-------------------------|------------|---------------|----------|  
| High-Value (Cluster 0) | $$$ (High)        | $$$ (High)   | Frequent    | 53%       |  
| Budget-Conscious (Cluster 1) | $ (Low)         | $ (Low)      | Infrequent | 47%       |  

### Key Insights  
- **High-Value Customers** are the backbone of revenue. They’re affluent, engaged, and responsive to campaigns.  
- **Budget-Conscious Customers** represent a retention opportunity—personalized campaigns might drive their spending higher.  

### Purchase Channel Preferences  
Customers in each cluster favor different purchase channels, with high-value customers engaging more across all touchpoints, including stores, catalogs, and online platforms.


## Tools & Techniques  
This project leverages a robust stack of tools to extract, analyze, and visualize insights:  
- **Data Analysis**: Pandas, NumPy  
- **Visualizations**: Matplotlib, Seaborn  
- **Clustering Algorithm**: Scikit-learn’s K-Means  
- **Feature Scaling**: StandardScaler  


## Final Thoughts  

This segmentation isn’t just about grouping customers—it’s about enabling smarter marketing strategies. Here’s how these findings can be applied:  
1. **Targeted Campaigns**:  
   - Reward **high-value customers** with loyalty perks and exclusive offers.  
   - Design affordable bundles or discounts to engage **budget-conscious customers**.  

2. **Channel Optimization**:  
   - Invest in the preferred channels of your high-value segment.  
   - Test promotional strategies to encourage budget-conscious customers to explore different channels.  

By identifying customer behavior at this granular level, iFood is poised to build stronger, more personalized relationships with its customers—and boost profitability along the way.  
