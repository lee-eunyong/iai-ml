import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X = 70 * np.random.rand(100, 1) # 100 random values between 0 and 100
y = 5 * X + 10 + np.random.randn(100, 1) * 10

# 2. Create and Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 3. Check Learned Parameters (Weights and Bias)
w1 = model.coef_[0][0] # Slope (Weight)
w0 = model.intercept_[0] # Intercept (Bias)

print(f" === Training Results === ")
print(f"Estimated Slope (w1): {w1 :.2f}")
print(f"Estimated Intercept (w0): {w0 :.2f}")
print(f"Final Equation: y = {w1: .2f}x + {w0 :.2f}")

# 4. Prediction and Evaluation
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\n === Model Evaluation === ")
print(f"Mean Squared Error (MSE): {mse :.2f}")
print(f"R-squared (R2 Score): {r2 :.2f}")

# 5. Visualization with English Labels
plt.figure(figsize=(10, 6))

# Plot actual data points
plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Data')

# Plot the learned regression line
plt.plot( X, y_pred, color='red', linewidth=2, label='Regression Line')

# Adding graph details in English
plt.title( "Linear Regression: Watcha Likes vs Audience", fontsize=14)
plt.xlabel( "Watcha 'Like' Count", fontsize=12)
plt.ylabel ( "Total Audience Count", fontsize=12)
plt. legend ()
plt.grid( True, linestyle='--', alpha=0.6)

# Display the plot
plt.show()