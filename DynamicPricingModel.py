import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# Load the data
products = pd.read_csv('/Users/mattobrien/Documents/TestDataV2/Enhanced_Products.csv')
orders = pd.read_csv('/Users/mattobrien/Documents/TestDataV2/Enhanced_Orders.csv')

# Data Cleaning
# Handling missing values
products['Stock'].fillna(products['Stock'].median(), inplace=True)
products['Price'].fillna(products['Price'].median(), inplace=True)
orders['Quantity'].fillna(0, inplace=True)  # Assuming missing quantity means no purchase

# Removing duplicates
products.drop_duplicates(inplace=True)
orders.drop_duplicates(inplace=True)

# Convert date to datetime for further manipulation and calculate monthly demand
orders['Date'] = pd.to_datetime(orders['Date'], errors='coerce')
orders['Month'] = orders['Date'].dt.month
monthly_demand = orders.groupby(['ProductID', 'Month'])['Quantity'].sum().reset_index()

# Merge the monthly demand and product data
full_data = products.merge(monthly_demand, on='ProductID', how='left')
full_data['Quantity'].fillna(0, inplace=True)  # Fill missing quantities with zero

# Calculate price elasticity and adjust stock values
full_data['Price_Elasticity'] = full_data['Price'] / full_data['Quantity'].clip(lower=1)
median_stock = full_data['Stock'].median()
full_data['Stock'] = full_data['Stock'].apply(lambda x: median_stock if x < 0 else x)

# Handling outliers and normalizing 'Price'
Q1 = full_data['Price'].quantile(0.25)
Q3 = full_data['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
full_data['Price'] = full_data['Price'].clip(lower=lower_bound, upper=upper_bound)

scaler = MinMaxScaler()
full_data['Normalized_Price'] = scaler.fit_transform(full_data[['Price']])

# Modeling
X = full_data[['Stock', 'Price_Elasticity', 'Quantity']]
y = full_data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the predicted vs actual prices
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.show()

# Price change analysis
full_data['Predicted_Price'] = model.predict(X)
full_data['Price_Change_Percentage'] = ((full_data['Predicted_Price'] - full_data['Price']) / full_data['Price']) * 100
product_example = full_data.iloc[0]
price_change_explanation = f"The model suggests a {product_example['Price_Change_Percentage']:.2f}% change in price for {product_example['Name']}, primarily due to changes in stock levels and demand."
print(price_change_explanation)

# Additional Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Distribution of Predicted Prices
plt.figure(figsize=(10, 5))
sns.histplot(full_data['Predicted_Price'], bins=20, kde=True)
plt.title('Distribution of Predicted Prices')
plt.xlabel('Predicted Price')
plt.ylabel('Frequency')
plt.show()

# Feature Importance Plot
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Include the additional metrics in the price change explanation
price_change_explanation = (
    f"The model suggests a {product_example['Price_Change_Percentage']:.2f}% change in price for {product_example['Name']}, "
    f"primarily due to changes in stock levels and demand.\n"
    f"Additional model evaluation metrics:\n"
    f"- Mean Absolute Error (MAE): {mae:.2f}\n"
    f"- R-squared (R²): {r2:.2f}"
)
print(price_change_explanation)
