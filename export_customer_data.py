import pandas as pd

# Load the original dataset
df = pd.read_csv('data/ecommerce_customer_data_large.csv')

# Replicate the feature engineering process from the notebook
df_prepared = df.copy()

# Handle missing values
df_prepared['Returns'] = df_prepared['Returns'].fillna(0).astype(int)

# Convert Purchase Date to datetime
df_prepared['Purchase Date'] = pd.to_datetime(df_prepared['Purchase Date'])

# Handle the duplicate Age columns properly
# The original data has both 'Customer Age' and 'Age' columns, we'll keep 'Customer Age' and drop the other Age column
df_prepared = df_prepared.drop(columns=['Age'])  # Drop the original 'Age' column first
df_prepared = df_prepared.rename(columns={
    'Customer Age': 'Age',  # Now rename 'Customer Age' to 'Age'
    'Customer ID': 'Customer_ID',
    'Customer Name': 'Customer_Name'
})

# Calculate product category preferences per customer
category_pivot = df_prepared.pivot_table(
    index='Customer_ID',
    columns='Product Category',
    values='Total Purchase Amount',
    aggfunc='sum',
    fill_value=0
)
category_pivot.columns = [f'Spent_on_{col}' for col in category_pivot.columns]

# Calculate payment method preferences per customer
payment_pivot = df_prepared.pivot_table(
    index='Customer_ID',
    columns='Payment Method',
    values='Total Purchase Amount',
    aggfunc='count',
    fill_value=0
)
payment_pivot.columns = [f'Used_{col.replace(" ", "_")}' for col in payment_pivot.columns]

# Do aggregations separately to avoid the DataFrame.name error
purchase_agg = df_prepared.groupby('Customer_ID').agg({
    'Total Purchase Amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
    'Quantity': ['sum', 'mean', 'std'],
    'Product Price': ['mean', 'std', 'min', 'max'],
    'Returns': ['sum', 'mean', 'max']
})
purchase_agg.columns = ['_'.join(col).strip() for col in purchase_agg.columns.values]

temporal_agg = df_prepared.groupby('Customer_ID')['Purchase Date'].agg(['min', 'max', 'count'])
temporal_agg.columns = ['First_Purchase_Date', 'Last_Purchase_Date', 'Purchase_Count_Check']

# Fixed: Use .name attribute instead of .rename()
age_df = df_prepared.groupby('Customer_ID')['Age'].mean()
age_df.name = 'Age'

gender_df = df_prepared.groupby('Customer_ID')['Gender'].apply(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
)
gender_df.name = 'Gender'

# Combine all aggregations
customer_features = purchase_agg.join(temporal_agg).join(age_df).join(gender_df).reset_index()

# Rename for clarity
customer_features = customer_features.rename(columns={
    'Total Purchase Amount_sum': 'Total_Lifetime_Value',
    'Total Purchase Amount_mean': 'Avg_Order_Value',
    'Total Purchase Amount_std': 'Order_Value_Volatility',
    'Total Purchase Amount_min': 'Min_Order_Value',
    'Total Purchase Amount_max': 'Max_Order_Value',
    'Total Purchase Amount_count': 'Total_Transactions',
    'Quantity_sum': 'Total_Items_Purchased',
    'Quantity_mean': 'Avg_Items_Per_Order',
    'Quantity_std': 'Items_Per_Order_Std',
    'Product Price_mean': 'Avg_Product_Price',
    'Product Price_std': 'Product_Price_Volatility',
    'Product Price_min': 'Min_Product_Price',
    'Product Price_max': 'Max_Product_Price',
    'Returns_sum': 'Total_Returns',
    'Returns_mean': 'Avg_Returns_Per_Order',
    'Returns_max': 'Max_Returns_Single_Order'
})

# Merge category and payment preferences
customer_features = customer_features.merge(category_pivot, on='Customer_ID', how='left')
customer_features = customer_features.merge(payment_pivot, on='Customer_ID', how='left')

# Define analysis date (latest date in dataset)
analysis_date = df_prepared['Purchase Date'].max()

# 1. RECENCY - Most important churn predictor!
customer_features['Days_Since_Last_Purchase'] = (
    analysis_date - customer_features['Last_Purchase_Date']
).dt.days

# 2. FREQUENCY - How often do they buy?
customer_features['Customer_Lifetime_Days'] = (
    customer_features['Last_Purchase_Date'] - customer_features['First_Purchase_Date']
).dt.days + 1  # Add 1 to avoid division by zero

customer_features['Purchases_Per_Month'] = (
    customer_features['Total_Transactions'] / (customer_features['Customer_Lifetime_Days'] / 30)
)

customer_features['Avg_Days_Between_Purchases'] = (
    customer_features['Customer_Lifetime_Days'] / 
    (customer_features['Total_Transactions'] + 1)
)

# 3. MONETARY - Spending patterns
customer_features['Spending_Per_Day'] = (
    customer_features['Total_Lifetime_Value'] / customer_features['Customer_Lifetime_Days']
)

# 4. TREND - Are they becoming more or less active? (VERY POWERFUL!)
def calculate_recent_vs_old_ratio(customer_id, recent_days=90):
    """Compare recent behavior vs historical behavior"""
    customer_data = df_prepared[df_prepared['Customer_ID'] == customer_id].copy()
    
    if len(customer_data) < 2:
        return 0  # Not enough data
    
    cutoff_date = analysis_date - pd.Timedelta(days=recent_days)
    
    recent = customer_data[customer_data['Purchase Date'] >= cutoff_date]
    old = customer_data[customer_data['Purchase Date'] < cutoff_date]
    
    if len(old) == 0:
        return 1  # New customer, recent activity only
    
    recent_purchases = len(recent)
    old_purchases = len(old)
    
    recent_days_active = (analysis_date - cutoff_date).days
    old_days_active = (cutoff_date - customer_data['Purchase Date'].min()).days
    
    if old_days_active == 0:
        return 1
    
    recent_rate = recent_purchases / recent_days_active
    old_rate = old_purchases / old_days_active
    
    if old_rate == 0:
        return 1
    
    return recent_rate / old_rate  # >1 = increasing activity, <1 = decreasing

customer_features['Activity_Trend_90d'] = customer_features['Customer_ID'].apply(
    lambda x: calculate_recent_vs_old_ratio(x, recent_days=90)
)

# 5. ENGAGEMENT METRICS
customer_features['Return_Rate'] = (
    customer_features['Total_Returns'] / customer_features['Total_Items_Purchased']
).fillna(0)

customer_features['Order_Value_Consistency'] = (
    customer_features['Order_Value_Volatility'] / (customer_features['Avg_Order_Value'] + 1)
).fillna(0)

# 6. CUSTOMER LIFECYCLE STAGE
customer_features['Is_New_Customer'] = (customer_features['Customer_Lifetime_Days'] <= 30).astype(int)
customer_features['Is_VIP'] = (
    customer_features['Total_Lifetime_Value'] > customer_features['Total_Lifetime_Value'].quantile(0.9)
).astype(int)

# 7. RECENT ACTIVITY FLAGS (Last 30/60/90 days)
for days in [30, 60, 90]:
    cutoff = analysis_date - pd.Timedelta(days=days)
    recent_activity = df_prepared[df_prepared['Purchase Date'] >= cutoff].groupby('Customer_ID').agg({
        'Total Purchase Amount': ['sum', 'count']
    }).reset_index()
    recent_activity.columns = ['Customer_ID', f'Spending_Last_{days}d', f'Purchases_Last_{days}d']
    customer_features = customer_features.merge(recent_activity, on='Customer_ID', how='left')
    customer_features[f'Spending_Last_{days}d'] = customer_features[f'Spending_Last_{days}d'].fillna(0)
    customer_features[f'Purchases_Last_{days}d'] = customer_features[f'Purchases_Last_{days}d'].fillna(0)

# Define churn threshold - adjust based on your business!
CHURN_THRESHOLD_DAYS = 90

customer_features['Churn'] = (
    customer_features['Days_Since_Last_Purchase'] > CHURN_THRESHOLD_DAYS
).astype(int)

# Encode categorical variables
customer_features['Gender'] = customer_features['Gender'].map({'Female': 1, 'Male': 0})

# Create customer_features_full (with Customer_ID for reference)
customer_features_full = customer_features.copy()

# 🛑 THE PREDICTIVE FIREWALL
# We exclude anything that "looks" into the 90-day churn window for the modeling dataset
exclude_cols = [
    'Customer_ID', 'First_Purchase_Date', 'Last_Purchase_Date', 
    'Days_Since_Last_Purchase', 'Purchase_Count_Check',
    'Purchases_Last_90d', 'Spending_Last_90d', 'Activity_Trend_90d',
    'Purchases_Last_60d', 'Spending_Last_60d',  # Also exclude 60-day window to be safe
    'Purchases_Last_30d', 'Spending_Last_30d'   # And 30-day window to be extra safe
]

if 'Customer_Name' in customer_features.columns:
    exclude_cols.append('Customer_Name')

feature_cols = [col for col in customer_features.columns 
                if col not in exclude_cols + ['Churn']]

# Create final modeling dataset (without Customer_ID)
df_final = customer_features[feature_cols + ['Churn']].fillna(0)

# Export both datasets
# 1. Export the modeling dataset (without Customer_ID)
df_final.to_csv(
    "data/data_after_cleaning_and_feature_enginiring.csv",
    index=False
)

# 2. Export the full dataset (with Customer_ID for reference)
customer_features_full.to_csv(
    "data/data_after_cleaning_and_feature_enginiring_with_customer_id.csv",
    index=False
)

print("Exported data/data_after_cleaning_and_feature_enginiring.csv (without Customer_ID)")
print("Exported data/data_after_cleaning_and_feature_enginiring_with_customer_id.csv (with Customer_ID)")