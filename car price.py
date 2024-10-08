
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv('car data.csv')

# Explore dataset
print(df.head())
print(df.info())  # Check for missing values and data types
# Drop rows with missing values (or handle them as needed)
df = df.dropna()

# One-Hot Encoding for categorical features (Brand, Model, etc.)
df = pd.get_dummies(df, drop_first=True)

# Check correlation to understand feature importance
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
# Define features (X) and target (y)
X = df.drop('Price', axis=1)  # Assuming 'Price' is the column to predict
y = df['Price']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_linear = linear_model.predict(X_test)
# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)
# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = linear_model.score(X_test, y_test)  # or rf_model.score(X_test, y_test) for RandomForest
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'R²: {r2}')

# Evaluate Linear Regression
print("Linear Regression Evaluation:")
evaluate_model(y_test, y_pred_linear)

# Evaluate Random Forest (if used)
print("Random Forest Evaluation:")
evaluate_model(y_test, y_pred_rf)
# Scatter plot for Linear Regression
plt.scatter(y_test, y_pred_linear)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.show()

# Scatter plot for Random Forest (optional)
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Random Forest)')
plt.show()