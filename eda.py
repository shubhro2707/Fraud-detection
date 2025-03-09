import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('processed_fraud_dataset.csv')

# Class Imbalance Analysis
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Fraud', palette='Set2')
plt.title('Fraud Class Distribution')
plt.show()

# Feature Distribution Analysis
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_features].hist(figsize=(12, 10), bins=20)
plt.suptitle('Feature Distributions', fontsize=16)
plt.show()

# Correlation Matrix Analysis
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Outlier Detection using Boxplot
plt.figure(figsize=(14, 6))
for feature in numeric_features:
    plt.figure()
    sns.boxplot(x='Fraud', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Fraud Status')
    plt.show()

# Feature Importance Analysis (Optional Step for Tree-Based Models)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Preparing data for modeling
X = df.drop('Fraud', axis=1)
y = df['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RandomForest Classifier for Feature Importance
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature Importance Plot
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()
