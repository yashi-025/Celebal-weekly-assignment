import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("personality_dataset.csv")
print(f"Columns Present in Datasets: \n{dataset.columns}")
print(f"\nShape of data: \n{dataset.shape}")
print(f"\n{dataset.info()}")
print(f"\nDescription of Dataset: \n{dataset.describe()}")

#missing values
print(f"\nMissing Values: \n{dataset.isnull().sum()}")
dataset.drop_duplicates(inplace=True)

#encoding
dataset['Stage_fear'] = dataset['Stage_fear'].map({'Yes': 1, 'No': 0})
dataset['Drained_after_socializing'] = dataset['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

#visualization
for col in dataset.select_dtypes(include=np.number).columns:
    sns.histplot(dataset[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()
    
sns.countplot(x='Stage_fear', hue='Personality', data=dataset)
plt.title("countplot of Stage_fear v/s Personality")
plt.show()

sns.scatterplot(x='Post_frequency', y='Social_event_attendance', hue='Personality', data= dataset)
plt.title("Post_frequency v/s Social_event_attendance v/s Personality")
plt.show()

num_df = dataset.drop(['Stage_fear','Drained_after_socializing','Personality'], axis=1)
corr = num_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("heatmap of Stage_fear v/s Drained_after_socializing v/s Personality")
plt.show()
