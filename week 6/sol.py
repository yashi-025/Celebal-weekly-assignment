import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = sns.load_dataset('titanic')

# Select relevant features
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

# Handle Missing Values
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

df['age'].fillna(df['age'].median(), inplace=True)

# Encode Categorical Variables
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define X and y
X = df.drop("survived", axis=1)
y = df["survived"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ModelEvaluator Class
class ModelEvaluator:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.metrics = {}

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        }
        print(f"\n---- {self.name} ----")
        print(classification_report(y_test, y_pred))
        return self.metrics

# Train and Evaluate Base Models
models = [
    (LogisticRegression(), "Logistic Regression"),
    (RandomForestClassifier(), "Random Forest"),
    (SVC(), "Support Vector Machine"),
    (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost")
]
results = {}
for model, name in models:
    evaluator = ModelEvaluator(model, name)
    evaluator.train(X_train, y_train)
    results[name] = evaluator.evaluate(X_test, y_test)

# Hyperparameter Tuning
# GridSearchCV - Random Forest
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='f1')
grid_rf.fit(X_train, y_train)

# RandomizedSearchCV - XGBoost
param_dist_xgb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
rand_xgb = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_distributions=param_dist_xgb,
    cv=5, scoring='f1', n_iter=5
)
rand_xgb.fit(X_train, y_train)

# Evaluate Tuned Models
print("\n===========GridSearchCV Tuned Random Forest=============")
tuned_rf_eval = ModelEvaluator(grid_rf.best_estimator_, "Tuned Random Forest")
tuned_rf_eval.train(X_train, y_train)
tuned_rf_eval.evaluate(X_test, y_test)

print("\n===============RandomizedSearchCV Tuned XGBoost==============")
tuned_xgb_eval = ModelEvaluator(rand_xgb.best_estimator_, "Tuned XGBoost")
tuned_xgb_eval.train(X_train, y_train)
tuned_xgb_eval.evaluate(X_test, y_test)

# Model Performance Summary
print("\n--- Model Performance Summary ---\n")
for name, metric in results.items():
    print(f"{name} -> F1: {metric['F1-Score']:.4f}")
