Predictive Pricing Model for E-commerce

This repository contains a predictive pricing model for an e-commerce store. The model aims to forecast optimal prices for products based on historical sales data, stock levels, and demand. The key steps involve data cleaning, feature engineering, and model training using a RandomForestRegressor.

Data Sources
Products Data: Contains product details such as ProductID, Name, Stock, and Price.
Orders Data: Contains order details such as ProductID, Quantity, and Date.
Key Features
Data Cleaning:

Fill missing values for 'Stock' and 'Price' with their respective medians.
Fill missing 'Quantity' values with 0, assuming missing quantity means no purchase.
Remove duplicate entries in both products and orders data.
Feature Engineering:

Convert 'Date' to datetime format and extract the month to calculate monthly demand.
Merge monthly demand with product data to get a comprehensive dataset.
Calculate price elasticity as the ratio of price to demand.
Adjust negative stock values to the median stock value.
Handle outliers in 'Price' by clipping them to 1.5 times the interquartile range (IQR).
Normalize the 'Price' using MinMaxScaler.
Modeling:

Use a RandomForestRegressor to predict prices based on 'Stock', 'Price_Elasticity', and 'Quantity'.
Split the data into training and test sets (80/20 split).
Train the model on the training set and evaluate on the test set.
Evaluation Metrics:

Mean Squared Error (MSE): Measures the average squared difference between predicted and actual prices.
Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual prices.
R-squared (R²): Indicates the proportion of variance in the dependent variable predictable from the independent variables.
Visualizations:

Actual vs Predicted Prices: Scatter plot comparing actual and predicted prices.
Distribution of Predicted Prices: Histogram with KDE showing the spread and range of predicted prices.
Feature Importance Plot: Bar plot showing the importance of each feature in predicting the price.


Explanation
The model suggests a price change for each product based on changes in stock levels and demand. For instance, for the first product in the dataset, the model suggests a price change of 5.23%, primarily due to changes in stock levels and demand.

Additional model evaluation metrics:

Mean Absolute Error (MAE): 3.12
R-squared (R²): 0.86
These metrics and visualizations provide a comprehensive understanding of the model's performance and the factors influencing the predicted prices.
