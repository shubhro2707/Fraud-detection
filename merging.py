# Step 1: Load Libraries
import pandas as pd

# Step 2: Load the Datasets
transaction_df = pd.read_csv("cust_transaction_details (1).csv")
customer_df = pd.read_csv("Customer_DF (1).csv")

# Step 3: Drop Irrelevant Columns
transaction_df.drop('Unnamed: 0', axis=1, inplace=True)
customer_df.drop('Unnamed: 0', axis=1, inplace=True)

# Step 4: Merge Datasets on 'customerEmail'
merged_df = pd.merge(transaction_df, customer_df, on='customerEmail', how='inner')

# Step 5: Save the Merged Dataset
merged_df.to_csv("merged_fraud_dataset.csv", index=False)

# Step 6: Display Information and Sample Data
print(merged_df.info())
print(merged_df.head())
