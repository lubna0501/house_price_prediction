import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
data = pd.read_csv(r"C:\Users\lubna\Downloads\archive5\train.csv")  # Using raw string literal

# Assuming the correct column names are 'GrLivArea' for square footage, 'BedroomAbvGr' for bedrooms, and 'FullBath' for bathrooms
# Replace these with the actual column names from your dataset
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']  # Assuming 'SalePrice' is the correct column name for the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save the trained model
model_file = "linear_regression_model.pkl"
joblib.dump(model, model_file)
print("Model saved successfully as", model_file)

