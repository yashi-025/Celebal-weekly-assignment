import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the Single Dataset
df = pd.read_csv("Housing.csv")
print("\nDataset shape:\n", df.shape)
print(df.head())

# Understand Target Column
target_column = 'price'  
y = df[target_column]
X = df.drop(columns=[target_column])

# Handle Missing Values by filling numeric column with median
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# Fill categorical columns with mode
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# One-hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Model Training and Evaluation
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_valid_scaled)
print("\nLinear Regression RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred_lr)))

# Ridge Regression
ridge = Ridge(alpha=10)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_valid_scaled)
print("\nRidge Regression RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred_ridge)))

# Lasso Regression
lasso = Lasso(alpha=0.001)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_valid_scaled)
print("\nLasso Regression RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred_lasso)))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_valid)
print("\nRandom Forest RMSE:", np.sqrt(mean_squared_error(y_valid, y_pred_rf)),"\n")

# Visualize Predictions
# corelation heatmap
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=['number']).corr()
top_corr_features = corr[target_column].abs().sort_values(ascending=False).head(10).index
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Top 10 Correlated Features with Price")
plt.show()

# target distribution
plt.figure(figsize=(8, 4))
sns.histplot(df[target_column], kde=True, bins=30, color='teal')
plt.title("Distribution of Target Variable (Price)")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Feature importance (Random forest)
importances = rf.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title("Top 15 Important Features (Random Forest)")
plt.show()

# residual plot
residuals = y_valid - y_pred_rf
plt.figure(figsize=(8, 4))
sns.histplot(residuals, kde=True, color='salmon')
plt.title("Distribution of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.show()

# actual v/s predicted scatter
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_valid, y=y_pred_rf, alpha=0.6, color='navy')
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Price (Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.show()
