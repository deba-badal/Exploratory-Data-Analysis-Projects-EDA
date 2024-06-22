import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
# Load the dataset
df = sns.load_dataset('iris')
# Display the first few rows of the dataset
print(df.head())

# Summary statistics
print(df.describe())

# Information about the dataset
print(df.info())
# Histograms for each feature
df.hist(bins=20, figsize=(10, 10))
plt.show()
# Box plots for each feature
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.drop(columns='species'))  # Exclude the 'species' column
plt.show()
# Scatter plot for each pair of features
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.show()
# Correlation matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap='coolwarm')  # Exclude the 'species' column
plt.show()
# Pair plot with species hue
sns.pairplot(df, hue='species')
plt.show()
# Check for missing values
print(df.isnull().sum())

# Identify outliers using z-score (excluding the 'species' column)
df_numeric = df.select_dtypes(include=[float, int])
z_scores = zscore(df_numeric)
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df_no_outliers = df[filtered_entries]
print(f"Original shape: {df.shape}, Shape after removing outliers: {df_no_outliers.shape}")
# Summarize the findings
print("Summary of EDA:")
print("1. Data Shape: ", df.shape)
print("2. Data Types: ", df.dtypes)
print("3. Summary Statistics: ", df.describe())
print("4. Missing Values: ", df.isnull().sum().sum())
print("5. Correlation Matrix: \n", df.drop(columns='species').corr())
