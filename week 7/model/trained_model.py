import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("dataset/Diamonds Prices2022.csv")

# Feature engineering
df = df[['carat', 'depth', 'table', 'price']]
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('model/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
