# Linear Relationship Analysis: Sales and Advertising Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns


# Import necessary libraries (already done above)
print("=== Sales and Advertising Linear Regression Analysis ===")
print("Step 1: Libraries imported successfully")

# Import/Create the data
print("\nStep 2: Creating sample Sales and Advertising dataset")

# Creating a sample dataset since no specific dataset was provided
np.random.seed(42)
n_samples = 100

# Generate advertising spend data (in thousands)
advertising_spend = np.random.uniform(10, 100, n_samples)

# Generate sales data with linear relationship + some noise
# Formula: Sales = 2.5 * Advertising + 15 + noise
noise = np.random.normal(0, 8, n_samples)
sales = 2.5 * advertising_spend + 15 + noise

# Create DataFrame
data = pd.DataFrame({
    'Advertising_Spend': advertising_spend,
    'Sales': sales
})

print("Dataset created with {} samples".format(n_samples))
print("Dataset shape:", data.shape)

# Analysis of data
print("\nStep 3: Data Analysis")
print("\nDataset Info:")
print(data.info())
print("\nBasic Statistics:")
print(data.describe())
print("\nFirst 10 rows:")
print(data.head(10))
print("\nCorrelation Matrix:")
correlation_matrix = data.corr()
print(correlation_matrix)
# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Declare feature variable and target variable
print("\nStep 4: Declaring feature and target variables")
X = data[['Advertising_Spend']]  # Feature variable (independent)
y = data['Sales']                # Target variable (dependent)

print("Feature variable (X) shape:", X.shape)
print("Target variable (y) shape:", y.shape)

# Plot scatter plot between X and y
print("\nStep 5: Creating scatter plot")
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(X, y, alpha=0.7, color='blue')
plt.xlabel('Advertising Spend (in thousands)')
plt.ylabel('Sales (in thousands)')
plt.title('Scatter Plot: Sales vs Advertising Spend')
plt.grid(True, alpha=0.3)

# Step 6: Checking and reshaping of X and y (if needed)
print("\nStep 6: Checking data shapes")
print("X shape before reshaping:", X.shape)
print("y shape before reshaping:", y.shape)

# X is already in the correct shape for sklearn
# y might need reshaping for some operations, but not for basic linear regression
print("Data shapes are appropriate for modeling")

# Step 7: Apply model
print("\nStep 7: Applying Linear Regression Model")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print("Model training completed")
print("Model coefficients:")
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Model equation
print(f"Linear Equation: Sales = {model.coef_[0]:.3f} * Advertising_Spend + {model.intercept_:.3f}")

# Step 8: Plot the Regression Line
print("\nStep 8: Plotting the Regression Line")

# Plot regression line on training data
plt.subplot(2, 2, 2)
plt.scatter(X_train, y_train, alpha=0.7, color='blue', label='Training Data')
plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Advertising Spend (in thousands)')
plt.ylabel('Sales (in thousands)')
plt.title('Linear Regression: Training Data')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot regression line on test data
plt.subplot(2, 2, 3)
plt.scatter(X_test, y_test, alpha=0.7, color='green', label='Test Data')
plt.scatter(X_test, y_pred_test, alpha=0.7, color='red', label='Predictions')
plt.xlabel('Advertising Spend (in thousands)')
plt.ylabel('Sales (in thousands)')
plt.title('Linear Regression: Test Data vs Predictions')
plt.legend()
plt.grid(True, alpha=0.3)




# Residual plot
plt.subplot(2, 2, 4)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.7, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Model Evaluation
print("\nModel Evaluation:")
print("Training Set Performance:")
train_r2 = r2_score(y_train, y_pred_train)
train_mse = mean_squared_error(y_train, y_pred_train)
print(f"R² Score: {train_r2:.4f}")
print(f"Mean Squared Error: {train_mse:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(train_mse):.4f}")

print("\nTest Set Performance:")
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print(f"R² Score: {test_r2:.4f}")
print(f"Mean Squared Error: {test_mse:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(test_mse):.4f}")




# Additional Analysis
print("\nAdditional Analysis:")

# Correlation coefficient
correlation = np.corrcoef(data['Advertising_Spend'], data['Sales'])[0, 1]
print(f"Correlation coefficient: {correlation:.4f}")

# Create a heatmap for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Prediction examples
print("\nPrediction Examples:")
sample_advertising = np.array([[20], [40], [60], [80]])
sample_predictions = model.predict(sample_advertising)



# Displaying of  predictions
for i, (ad_spend, predicted_sales) in enumerate(zip(sample_advertising.flatten(), sample_predictions)):
    print(f"Advertising Spend: ${ad_spend}k → Predicted Sales: ${predicted_sales:.2f}k")

print("\n=== Analysis Complete ===")
print("\nKey Findings:")
print(f"1. Strong positive linear relationship (r = {correlation:.3f})")
print(f"2. Model explains {test_r2:.1%} of variance in sales")
print(f"3. For every $1k increase in advertising, sales increase by approximately ${model.coef_[0]:.2f}k")
print(f"4. Base sales (when advertising = 0) is approximately ${model.intercept_:.2f}k")