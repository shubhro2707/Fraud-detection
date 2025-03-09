import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the cleaned dataset
file_path = 'merged_fraud_dataset.csv'
merged_df = pd.read_csv(file_path)

# Drop non-numeric and irrelevant columns
merged_df.drop(['customerEmail', 'transactionId', 'orderId', 'paymentMethodId', 
                 'customerPhone', 'customerDevice', 'customerIPAddress', 'customerBillingAddress'], 
                 axis=1, inplace=True)

# Handle missing values
merged_df.fillna({'paymentMethodType': merged_df['paymentMethodType'].mode()[0],
                  'orderState': merged_df['orderState'].mode()[0]}, inplace=True)

# Feature Scaling and Encoding
numeric_features = ['transactionAmount', 'No_Transactions', 'No_Orders', 'No_Payments']
categorical_features = ['paymentMethodProvider', 'orderState']

# Define Transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

# Create Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Apply Transformation
transformed_data = preprocessor.fit_transform(merged_df)

# Create a new DataFrame with transformed data
encoded_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
processed_df = pd.DataFrame(transformed_data, columns=numeric_features + list(encoded_columns))

# Add back the target column
processed_df['Fraud'] = merged_df['Fraud'].values

# Save the processed dataset
processed_df.to_csv('processed_fraud_dataset.csv', index=False)

print("Data processing completed and saved as 'processed_fraud_dataset.csv'")