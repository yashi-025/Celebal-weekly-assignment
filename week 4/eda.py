import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Load dataset
dataset = sns.load_dataset("titanic")

#Data Overview and summary
print(dataset.head())
print("\nShape of dataset: ", dataset.shape)
print("\nColumns present in dataset:\n", dataset.columns)
print("\nInfo about dataset:\n", dataset.info())
print("\nData types:\n", dataset.dtypes)
print("\nOverview of dataset:\n", dataset.describe())

#Missing values
print("\nMissing values in dataset:\n", dataset.isnull().sum())

#Visualize missing data on heatmap
plt.figure(figsize = (10, 6))
sns.heatmap(dataset.isnull(), cbar = False, cmap = 'viridis', yticklabels = False)
plt.title("Missing Values Heatmap")
plt.show()

#Distribution of Numerical features
num_cols = dataset.select_dtypes(include= [np.number]).columns.tolist()
for col in num_cols:
    plt.figure(figsize = (8,6))
    sns.histplot(dataset[col].dropna(),kde = True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("frequency")
    plt.show()
    
#Boxplot for Outlier Detection
for col in num_cols:
    plt.figure(figsize= (8,6)) 
    sns.boxplot(x = dataset[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.show()
    
# Correlation Matrix and Heatmap
plt.figure(figsize=(10, 6))
corr = dataset[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.show()

# Categorical vs Survival (Relationships)
categorical_cols = ['sex', 'pclass', 'embarked', 'who', 'deck', 'class']
for col in categorical_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(data = dataset, x=col, hue="survived")
    plt.title(f"Survival Count by {col}")
    plt.xticks(rotation=45)
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.show()

# Age vs Fare Colored by Survival
plt.figure(figsize=(8, 6))
sns.scatterplot(data = dataset, x='age', y='fare', hue='survived', palette='Set1')
plt.title("Age vs Fare by Survival")
plt.show()

# Pairplot (for selected features)
selected_cols = ['survived', 'age', 'fare', 'pclass']
sns.pairplot(dataset[selected_cols], hue='survived', palette='husl')
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()