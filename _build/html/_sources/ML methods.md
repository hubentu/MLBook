# **Popular Machine Learning Methods**

In this section, we'll cover the core machine learning methods, including **regression**, **classification**, **dimensionality reduction**, and **gradient boosting** algorithms. For each method, I'll provide a practical project idea along with a brief outline of how to solve it using these techniques.

---

## **1. Regression Algorithms**

### **Overview**:
- **Regression** is used to predict a continuous value based on one or more input features.
- **Common Algorithms**: Linear Regression, Ridge Regression, Lasso Regression, Polynomial Regression.

### **Project 1: House Price Prediction**
**Goal**: Predict the price of a house based on features such as the size of the house, number of bedrooms, location, and age.

**Dataset**: Use the **Kaggle House Prices dataset** (or any open dataset on house prices).

**Solution**:
1. **Data Preprocessing**:
   - Handle missing values.
   - One-hot encode categorical variables (e.g., neighborhood, house style).
   - Normalize numerical features like square footage and lot size.
2. **Feature Engineering**:
   - Create derived features, such as **price per square foot** or **age of the house**.
3. **Modeling**:
   - Start with **Linear Regression** and observe performance.
   - Experiment with **Ridge Regression** and **Lasso Regression** for regularization to prevent overfitting.
   - If the data is non-linear, try **Polynomial Regression**.
4. **Evaluation**:
   - Use **Mean Squared Error (MSE)** and **R-squared** as performance metrics.
   - Use **cross-validation** to check the model's robustness.

**Sample Code**:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Load dataset and preprocess it
# Assume df is a pandas dataframe with processed features and target

X = df.drop('price', axis=1)  # Features
y = df['price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
```

---

## **2. Classification Algorithms**

### **Overview**:
- **Classification** is used to predict a categorical label based on input features.
- **Common Algorithms**: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Trees, Random Forest.

### **Project 2: Customer Churn Prediction**
**Goal**: Predict whether a customer will churn (leave) a subscription-based service based on usage patterns, customer support interactions, and demographic data.

**Dataset**: Use the **Kaggle Telecom Customer Churn dataset**.

**Solution**:
1. **Data Preprocessing**:
   - One-hot encode categorical variables like **contract type**, **payment method**, etc.
   - Scale continuous variables like **tenure**, **monthly charges**, and **total charges**.
2. **Modeling**:
   - Start with **Logistic Regression** as a baseline.
   - Try **Random Forest** for a more robust, non-linear approach.
   - Use **SVM** for high-dimensional feature spaces and **KNN** for instance-based learning.
3. **Feature Engineering**:
   - Derive features such as **average monthly charges per year of tenure**.
4. **Evaluation**:
   - Use metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
   - Use **ROC-AUC** for assessing classification performance.

**Sample Code**:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataset and preprocess it
# Assume df is a pandas dataframe with processed features and target

X = df.drop('churn', axis=1)  # Features
y = df['churn']  # Target (binary)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("Accuracy: ", accuracy)
print("ROC-AUC: ", roc_auc)
```

---

## **3. Dimensionality Reduction Algorithms**

### **Overview**:
- **Dimensionality Reduction** is used to reduce the number of features while retaining important information.
- **Common Algorithms**: Principal Component Analysis (PCA), t-SNE, Linear Discriminant Analysis (LDA), Autoencoders (Neural Networks).

### **Project 3: Visualizing High-Dimensional Data (MNIST Handwritten Digits)**
**Goal**: Use dimensionality reduction techniques to reduce the dimensionality of the **MNIST dataset** and visualize the data in 2D.

**Dataset**: MNIST dataset (available via **scikit-learn** or **Keras**).

**Solution**:
1. **Load the MNIST Dataset**.
2. **Apply PCA** to reduce the number of dimensions (e.g., from 784 to 50).
3. **Use t-SNE** to visualize the data in 2D space, capturing local similarities between the digits.
4. **Plot the results** using a scatter plot.

**Sample Code**:
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load dataset (MNIST)
digits = load_digits()
X = digits.data  # Features (64-dimensional images)
y = digits.target  # Labels

# Apply PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Apply t-SNE to reduce dimensions to 2D for visualization
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_pca)

# Visualize the 2D projection
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.colorbar()
plt.title('t-SNE Visualization of MNIST Digits')
plt.show()
```

---

## **4. Gradient Boosting Algorithms**

### **Overview**:
- **Gradient Boosting** is an ensemble learning technique that combines the predictions of several weak learners to produce a more accurate model.
- **Common Algorithms**: XGBoost, LightGBM, CatBoost.

### **Project 4: Predicting Loan Default using XGBoost**
**Goal**: Predict whether a customer will default on a loan based on credit score, income, loan amount, and other financial data.

**Dataset**: Use any financial dataset with customer loan data (e.g., Kaggle's **Home Credit Default Risk** dataset).

**Solution**:
1. **Data Preprocessing**:
   - Handle missing values and encode categorical features using **one-hot encoding** or **target encoding**.
   - Scale numeric features.
2. **Modeling**:
   - Use **XGBoost** for predicting loan defaults. XGBoost is known for its efficiency and performance on tabular data.
3. **Feature Importance**:
   - After training, use **XGBoost’s feature importance** to identify the most important features influencing the model's predictions.
4. **Evaluation**:
   - Use **cross-validation** to ensure the model generalizes well.
   - Evaluate performance using **accuracy**, **precision**, **recall**, and **ROC-AUC**.

**Sample Code**:
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataset and preprocess it
# Assume df is a pandas dataframe with processed features and target

X = df.drop('default', axis=1)  # Features
y = df['default']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("Accuracy: ", accuracy)
print("ROC-AUC: ", roc_auc)
```
