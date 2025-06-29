import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
dataset = pd.read_csv("Smartphones_cleaned_dataset.csv")
print(f"Columns Present in Datasets: \n{dataset.columns.tolist()}")
print(f"\nShape of data: \n{dataset.shape}")
print(f"\n{dataset.info()}")
print(f"\nDescription of Dataset: \n{dataset.describe()}")

#missing values
print(f"\nMissing Values: \n{dataset.isnull().sum()}")
dataset.drop_duplicates(inplace=True)

#encoding
for col in dataset.select_dtypes(include = ['bool']).columns:
    dataset[col] = dataset[col].astype(int)
    #another method: dataset[col] = dataset[col].map({True: 1, False: 0})
# to get selected datatype column only    
print(dataset.select_dtypes(include = ['int']).head(6))

#visualization
for col in dataset.select_dtypes(include=np.number).columns:
    sns.histplot(dataset[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()
    
#correlation matrix for numerical column help to analyze relation b/w features
plt.figure(figsize=(12, 8))
correlation = dataset.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Brand distribution
plt.figure(figsize=(12, 6))
sns.boxplot(data=dataset, x='brand_name', y='price')
plt.xticks(rotation=45)
plt.title("Price Distribution by Brand")
plt.tight_layout()
plt.show()

#Brand summary
brand_summary = dataset.groupby('brand_name')[['price', 'battery_capacity', 'ram_capacity']].agg(['mean', 'max', 'min'])
print(brand_summary)