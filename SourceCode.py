# 1. Load and Understand the Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

print(df.head())
print("Shape:", df.shape)
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())


# 2. Exploratory Data Analysis (EDA)
plt.figure(figsize=(6,4))
sns.histplot(df['MedHouseVal'], bins=30, kde=True)
plt.title("Distribution of Median House Value")
plt.show()

# Scatter plot:
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df)
plt.title("Median Income vs House Value")
plt.show()

# Correlation heatmap:
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Boxplot:
df['IncomeBin'] = pd.qcut(df['MedInc'], q=4)
plt.figure(figsize=(8,5))
sns.boxplot(x='IncomeBin', y='MedHouseVal', data=df)
plt.title("House Value by Income Quartile")
plt.xticks(rotation=45)
plt.show()

# Visual outlier detection
plt.figure(figsize=(6,4))
sns.boxplot(x=df['MedHouseVal'])
plt.title("Boxplot of Median House Value")
plt.show()


# 3. Data Preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Select features
features = ['MedInc', 'AveRooms', 'AveOccup', 'HouseAge', 'AveBedrms']
X = df[features]
y = df['MedHouseVal']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

# 4 & 5. Build and Train Two Regression Models

from sklearn.linear_model import LinearRegression

# Model A: Simple Linear Regression (using only MedInc)
X_simple = df[['MedInc']]
X_simple_scaled = scaler.fit_transform(X_simple)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple_scaled, y, test_size=0.2, random_state=42)

model_a = LinearRegression()
model_a.fit(X_train_s, y_train_s)

# Model B: Multiple Linear Regression
model_b = LinearRegression()
model_b.fit(X_train, y_train)


# 6. Evaluate the Models

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    print(f"\n{name} Evaluation:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))
    return y_pred

y_pred_a = evaluate(model_a, X_test_s, y_test_s, "Simple Linear Regression")
y_pred_b = evaluate(model_b, X_test, y_test, "Multiple Linear Regression")


# 8. Required Visualizations

# Scatter plot + regression line for simple regression
plt.figure(figsize=(6,4))
sns.regplot(x=X_test_s.flatten(), y=y_test_s, line_kws={"color": "red"})
plt.title("Simple Regression: MedInc vs House Value")
plt.xlabel("Median Income (scaled)")
plt.ylabel("House Value")
plt.show()

# Actual vs Predicted plot for multiple regression
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred_b)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Multiple Regression)")
plt.show()

# Residual plot
residuals = y_test - y_pred_b
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred_b, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()
