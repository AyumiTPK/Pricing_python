import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Competitor data:
num_cars = np.array([150, 300, 600, 1000]).reshape(-1, 1)
competitor_prices = np.array([379, 479, 579, 749])

# Initialize and fit the regression model
model = LinearRegression()
model.fit(num_cars, competitor_prices)

# Predict prices
car_tiers = np.array([100, 200, 500, 700]).reshape(-1, 1)
predicted_prices = model.predict(car_tiers)

# Output the predicted prices for each tier
for i, tier in enumerate(car_tiers):
    print(f"Optimal price for {tier[0]} cars: ${predicted_prices[i]:.2f}")

# Plot the results
plt.scatter(num_cars, competitor_prices, color='blue', label='Competitor Prices')
plt.plot(car_tiers, predicted_prices, color='red', label='Predicted Prices')
plt.xlabel('Number of Cars')
plt.ylabel('Price ($)')
plt.title('Competitor Pricing vs Predicted Pricing')
plt.legend()
plt.show()
