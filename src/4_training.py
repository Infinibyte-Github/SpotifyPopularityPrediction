"""
training of the model on the scaled dataset
Author: Bart Schelpe
Filename: 4_Training.py
Dataset: 🎹 Spotify Tracks Dataset
Link: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
Code based on: Python4AI PowerPoint presentations and documentation of varying Python packages
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the scaled datasets
dfTrain = pd.read_pickle('../data/dfTrainMinMaxScaler.pickle')
dfTest = pd.read_pickle('../data/dfTestMinMaxScaler.pickle')

# Split the datasets into features and target
X_train = dfTrain.drop(columns=['popularity'])
y_train = dfTrain['popularity']

X_test = dfTest.drop(columns=['popularity'])
y_test = dfTest['popularity']

# # Initialize and train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Initialize the SVR model
# svm_model = SVR(kernel='rbf', verbose=True)  # You can choose different kernels like 'linear', 'poly', 'rbf', etc.
#
# # Train the SVR model on the training data
# svm_model.fit(X_train, y_train)
#
# # Make predictions on the test data
# y_pred = svm_model.predict(X_test)

# Initialize the KNN classifier
knn_classifier = KNeighborsRegressor(n_neighbors=3)  # You can choose the number of neighbors (k) here

# Train the KNN classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Evaluate the model)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot actual vs. predicted values
Maxplot = max(max(y_test), max(y_pred))
Minplot = min(min(y_test), min(y_pred))
plt.plot([Minplot, Maxplot], [Minplot, Maxplot], color='red')
plt.scatter(y_test, y_pred, color="red")
plt.axline((0, 0), slope=1)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.show()

# error_rate = []  # Will take some time
# for i in range(1, 40):
#     knn = KNeighborsRegressor(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.sqrt(mean_squared_error(y_test, pred_i)))
#     print(f"K = {i} -> Error = {error_rate[-1]}")
#
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
#          markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()
