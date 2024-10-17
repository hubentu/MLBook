---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


## Example
Let’s walk through a practical example of using machine learning to build a predictive model. We’ll follow a structured approach that covers the entire machine learning pipeline, including:

1. **Problem Definition**
2. **Data Exploration (EDA)**
3. **Data Cleaning**
4. **Feature Engineering**
5. **Model Selection**
6. **Model Evaluation**
7. **Model Tuning**
8. **Deployment and Monitoring**

We'll use a fictitious problem to keep things straightforward and generalizable. Suppose we are working on predicting **whether a customer will churn** (i.e., stop using the service) based on customer data for a telecom company.

---

### **Problem Definition**
**Objective:** Predict whether a customer will churn (yes/no) based on customer data such as demographics, usage patterns, and service history.

**Data:** You are provided with a dataset containing customer features:
- **CustomerID**: Unique identifier for each customer.
- **Tenure**: How long the customer has been with the company.
- **MonthlyCharges**: The amount billed to the customer monthly.
- **TotalCharges**: The total amount billed.
- **Contract**: The type of contract (e.g., month-to-month, one year).
- **PaymentMethod**: How the customer pays (e.g., credit card, bank transfer).
- **Churn**: The target variable (1 for churn, 0 for no churn).

---

### **Step 1: Data Exploration (EDA)**

We begin by loading and exploring the data.

```{code-cell} python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/TelecomCustomerChurn.csv')

# Display first few rows
print(data.head())
```

**Initial Checks:**
- **Summary statistics** of numerical features: mean, median, and distribution of each variable.
  
```python
# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())
```

- **Class imbalance:** Check the distribution of the target variable (`Churn`). Imbalanced classes could affect model performance.
  
```python
# Check class distribution
print(data['Churn'].value_counts())
sns.countplot(x='Churn', data=data)
plt.show()
```

**Visualizing Correlations:**
Visualize the correlation between numerical features (e.g., tenure, monthly charges) and churn to detect potential patterns.

```python
# Correlation matrix
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```

Insights from EDA:
- If the target variable `Churn` is imbalanced (e.g., 80% no churn, 20% churn), we might need to address this during model training.
- Monthly charges and tenure could show strong correlation with churn, providing clues for feature importance.

---

### **Step 2: Data Cleaning**

We now handle missing data, incorrect formats, and other issues.

```python
# Handle missing values in 'TotalCharges'
# 'TotalCharges' might have some empty strings that need to be converted to NaN and filled.
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing values in 'TotalCharges' with the median
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
```

**Categorical Variable Encoding:**
For categorical features like `Contract` and `PaymentMethod`, convert them to numerical format using one-hot encoding or label encoding.

```python
# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Contract', 'PaymentMethod'], drop_first=True)
```

---

### **Step 3: Feature Engineering**

Next, we generate new features and modify existing ones to improve the model’s ability to predict churn.

1. **Customer Tenure Grouping:**
   Customers with longer tenures might behave differently, so we can bucket the `Tenure` feature into categories.
   
```python
# Create tenure groups
bins = [0, 12, 24, 36, 48, 60, 72]
labels = ['0-12', '12-24', '24-36', '36-48', '48-60', '60-72']
data['TenureGroup'] = pd.cut(data['Tenure'], bins=bins, labels=labels)
```

2. **Interaction Features:**
   Interaction between features like `MonthlyCharges` and `Contract` type might provide insight into churn behavior.
   
```python
# Interaction term between MonthlyCharges and Contract type
data['Charges_Contract'] = data['MonthlyCharges'] * data['Contract_One year']
```

3. **Log Transformation:**
   If `TotalCharges` is highly skewed, applying a log transformation can make the distribution more normal and improve model performance.
   
```python
import numpy as np
data['Log_TotalCharges'] = np.log1p(data['TotalCharges'])
```

---

### **Step 4: Model Selection**

We now split the data into training and test sets and select candidate models to compare.

```python
from sklearn.model_selection import train_test_split

# Split data into training and test sets
X = data.drop(['Churn', 'CustomerID'], axis=1)  # Features
y = data['Churn']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Model Choices:**
We'll test multiple models, including:
- **Logistic Regression** (baseline)
- **Random Forest**
- **Gradient Boosting (XGBoost or LightGBM)**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Initialize models
log_reg = LogisticRegression()
rf = RandomForestClassifier()
xgb = XGBClassifier()

# Fit models
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
```

---

### **Step 5: Model Evaluation**

Evaluate the performance of each model using relevant metrics.

```python
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Make predictions
log_reg_pred = log_reg.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

# Accuracy
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_reg_pred)}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred)}")
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred)}")

# Classification Report
print(classification_report(y_test, xgb_pred))

# ROC-AUC Score
print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, xgb_pred)}")
```

For imbalanced datasets, focus on metrics like precision, recall, F1 score, and ROC-AUC rather than accuracy alone.

**Confusion Matrix:**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix for XGBoost
cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
```

---

### **Step 6: Model Tuning**

Once we have a good model, we perform hyperparameter tuning using techniques like **Grid Search** or **Random Search**.

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1]
}

# Perform grid search
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters
print(grid_search.best_params_)
```

---

### **Step 7: Handling Imbalanced Data**

Since churn is likely an imbalanced problem, we can address this by:

1. **Oversampling the minority class** (using SMOTE):
   
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
```

2. **Class weighting** (in RandomForestClassifier or XGBoost):
   
```python
rf = RandomForestClassifier(class_weight='balanced')
xgb = XGBClassifier(scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)))  # For imbalanced datasets
```

---

### **Step 8: Model Deployment and Monitoring**

Once the model is trained and tested, we package it for deployment. Using a framework like **Flask** or **FastAPI**, we can serve the model predictions through an API.

```python
from flask import Flask, request, jsonify
import pickle

# Load the trained model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run()
```

Finally, we monitor the model in production using key performance indicators (KPIs) such as accuracy drift or changes in the distribution of incoming data, ensuring the model continues to perform well.

---

### Conclusion



This pipeline outlines a complete approach for building a machine learning model, from data exploration to deployment. It demonstrates a real-world application that includes model tuning, handling imbalanced data, and ensuring deployment-ready code. This structure is typical of how a machine learning engineer would approach building a model end-to-end in a professional setting.