import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("Advertising.csv")

# Show dataset
print("First 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

# Features and target
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Predict sales for new advertising budget
new_data = pd.DataFrame({
    "TV": [100],
    "Radio": [25],
    "Newspaper": [20]
})

prediction = model.predict(new_data)

print("\nPredicted Sales:", prediction[0])

# Actual vs Predicted graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()