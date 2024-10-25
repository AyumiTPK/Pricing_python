import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Pricing data
data = {
    "G": [
        (10, 249), (50, 249), (100, 249), (200, 249), (300, 249), (500, 249), (700, 249)
    ],
    "I": [
        (10, 379), (50, 379), (100, 379), (200, 479), (300, 479), (500, 579), (700, 749)
    ],
    "W1": [
        (10, 35.36), (50, 141.99), (100, None), (200, None), (300, None), (500, None), (700, None)
    ],
    "W2": [
        (10, 31.8), (50, 124.22), (100, None), (200, None), (300, None), (500, None), (700, None)
    ]
}

# Extract vehicle quantities and prices, excluding None values
vehicle_quantities = []
prices = []

for company, values in data.items():
    for quantity, price in values:
        if price is not None:
            vehicle_quantities.append(quantity)
            prices.append(price)

# Convert lists to numpy arrays for regression
num_cars = np.array(vehicle_quantities).reshape(-1, 1)
competitor_prices = np.array(prices)

# Initialize and fit the regression model
model = LinearRegression()
model.fit(num_cars, competitor_prices)

# Predict prices
car_tiers = np.array([100, 200, 500, 700]).reshape(-1, 1)
predicted_prices = model.predict(car_tiers)

# Set pricing to be cheaper or equal to competitors
for i, tier in enumerate(car_tiers):
    competitor_price = competitor_prices[num_cars.flatten() == tier].min()
    predicted_prices[i] = min(predicted_prices[i], competitor_price)

# Output the adjusted prices for each tier
for i, tier in enumerate(car_tiers):
    print(f"Adjusted price for {tier[0]} cars: ${predicted_prices[i]:.2f}")

# Plot the results
plt.scatter(num_cars, competitor_prices, color='blue', label='Competitor Prices')
plt.plot(car_tiers, predicted_prices, color='red', label='Predicted Prices')
plt.xlabel('Number of Cars')
plt.ylabel('Price ($)')
plt.title('Competitor Pricing vs Predicted Pricing')
plt.legend()
plt.show()
