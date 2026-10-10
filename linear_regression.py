import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs('output', exist_ok=True)

def train_regression():
    np.random.seed(42)
    years_experience = np.random.uniform(1, 10, 40).reshape(-1, 1)
    salary = 30000 + (years_experience * 9500) + np.random.normal(0, 3000, (40, 1))

    X_train, X_test, y_train, y_test = train_test_split(years_experience, salary, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("--- Model Accuracy & Metrics ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: ")
    print(f"Regression Line Equation: Salary = {model.intercept_[0]:.2f} + ({model.coef_[0][0]:.2f} * YearsExperience)")

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color='blue', label='Train Data')
    plt.scatter(X_test, y_test, color='green', label='Test Data')
    
    X_range = np.linspace(1, 10, 100).reshape(-1, 1)
    plt.plot(X_range, model.predict(X_range), color='red', linewidth=2, label='Regression Line')

    plt.xlabel('Years of Experience')
    plt.ylabel('Salary ($)')
    plt.title('Simple Linear Regression: Salary vs Experience')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('output/regression_line.png')
    plt.close()

    print("\nPlot saved successfully to output/regression_line.png")

if __name__ == "__main__":
    train_regression()
