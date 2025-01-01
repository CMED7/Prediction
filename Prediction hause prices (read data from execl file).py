import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read data from an Excel file
df = pd.read_excel('D:/data/ /BEN TALEB MOHAMED/troisieme annee faculte/ml/Nouveau Feuille de calcul Microsoft Excel.xlsx')
#df = pd.read_excel('C:/Users/Lenovo/Desktop/Nouveau dossier (4)/Nouveau Feuille de calcul Microsoft Excel.xlsx')

# Separate features (X) and target variable (y)
X = df[['Square Footage']]
y = df['Price']

# Separate features (X) and target variable (y)
X = df[['Square Footage']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize the model (optional)
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, model.predict(X_train), color='red')
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("Linear Regression Model")
plt.show()
