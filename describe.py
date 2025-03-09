# Step 1: Load Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 2: Load the Merged Dataset
merged_df = pd.read_csv("merged_fraud_dataset.csv")

# Step 3: Inspect Data Types and Unique Values
print(merged_df.dtypes)
print(merged_df.nunique())

# Step 4: Handling Missing Values
print("Missing Values:\n", merged_df.isnull().sum())

# Step 5: Identify Non-Numeric Columns
non_numeric_cols = merged_df.select_dtypes(include=['object']).columns.tolist()
print("Non-Numeric Columns:\n", non_numeric_cols)

# Step 6: Data Cleaning
# Convert boolean column 'Fraud' to 0/1
merged_df['Fraud'] = merged_df['Fraud'].astype(int)

# Encode categorical columns with meaningful mappings
category_mappings = {
    'paymentMethodType': {"card": 0, "paypal": 1, "apple_pay": 2},
    'orderState': {"completed": 0, "failed": 1, "pending": 2},
}

for col, mapping in category_mappings.items():
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].map(mapping)

# Step 7: Remove Irrelevant Columns
columns_to_drop = ['customerEmail', 'transactionId', 'orderId', 'customerPhone',
                    'customerDevice', 'customerIPAddress', 'customerBillingAddress']
merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Step 8: Confirm Cleaned Data
print("Cleaned Data Preview:\n", merged_df.head())
print("Data Types After Cleaning:\n", merged_df.dtypes)

# Step 9: Save Cleaned Dataset (Optional)
# merged_df.to_csv("/mnt/data/cleaned_fraud_dataset.csv", index=False)

print("Data Cleaning Completed.")
