import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
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

# Create a polynomial regression model
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit the polynomial model
poly_model.fit(num_cars, competitor_prices)

# Predict prices using polynomial regression
car_tiers = np.array([100, 200, 500, 700]).reshape(-1, 1)
predicted_prices_poly = poly_model.predict(car_tiers)

# Output the predicted prices
for i, tier in enumerate(car_tiers):
    print(f"Optimal price for {tier[0]} cars (polynomial): ${predicted_prices_poly[i]:.2f}")

# Plot the polynomial results
plt.scatter(num_cars, competitor_prices, color='blue', label='Competitor Prices')
plt.plot(car_tiers, predicted_prices_poly, color='green', label='Polynomial Predicted Prices')
plt.xlabel('Number of Cars')
plt.ylabel('Price ($)')
plt.title('Competitor Pricing vs Polynomial Predicted Pricing')
plt.legend()
plt.show()
