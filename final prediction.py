import joblib

# Load the trained model
model_file = "linear_regression_model.pkl"
model = joblib.load(model_file)

# Define a function to make predictions
def predict_house_price(model):
    # Accept user input for square footage, bedrooms, and bathrooms
    square_footage = float(input("Enter the square footage of the house: "))
    bedrooms = int(input("Enter the number of bedrooms: "))
    bathrooms = int(input("Enter the number of bathrooms: "))
    
    # Prepare the input data for prediction and reshape it to a 2D array
    input_data = [[square_footage, bedrooms, bathrooms]]
    
    # Make the prediction
    predicted_price = model.predict(input_data)
    
    return predicted_price[0]  # Return the predicted price

# Function to format the predicted price in crore, lakh, and thousand formats
def format_price(price):
    crore = int(price) // 10000000
    lakh = (int(price) % 10000000) // 100000
    thousand = (int(price) % 100000) // 1000
    return crore, lakh, thousand

# Make predictions based on user input
predicted_price = predict_house_price(model)
crore, lakh, thousand = format_price(predicted_price)
print(f"Predicted price for the house is: {crore} crore {lakh} lakh {thousand} thousand")
