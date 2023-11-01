import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('MSFT.csv')
print(df.describe())

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.dropna(inplace=True)

# Visualize the closing prices
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'])
plt.title('Historical Stock Prices of MSFT')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

# Create a new column for the target variable (e.g., next day's closing price)
df['target'] = df['Close'].shift(-1)

# Drop any remaining rows with missing values
df.dropna(inplace=True)

df['20MA'] = df['Close'].rolling(window=20).mean()
df['50MA'] = df['Close'].rolling(window=50).mean()

# Plotting the moving averages
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Closing Price')
plt.plot(df.index, df['20MA'], label='20-day Moving Average')
plt.plot(df.index, df['50MA'], label='50-day Moving Average')
plt.legend()
plt.title('Moving Averages for Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Split the data into features and target
X = df[['Open', 'Low', 'High', 'Volume']]
y = df['target']

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = X_scaled + np.random.normal(0, 0.075, X_scaled.shape)
y = y + np.random.normal(0, 0.01, y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Visualize the predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test.index, y_test.values, label='Actual',color='red')
plt.scatter(y_test.index, y_pred, label='Predicted',alpha=0.5,color='black')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Reshape the data for RNN
X_rnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split the data into training and testing sets for RNN
X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_rnn, y, test_size=0.2, random_state=0)

# Create an RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(50, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]), activation='relu'))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(X_train_rnn, y_train_rnn, epochs=50, batch_size=32, verbose=2)

# Make predictions using RNN
y_pred_rnn = rnn_model.predict(X_test_rnn)

# Model evaluation for RNN
rnn_mse = mean_squared_error(y_test_rnn, y_pred_rnn)
rnn_mae = mean_absolute_error(y_test_rnn, y_pred_rnn)
print(f"RNN Mean Squared Error: {rnn_mse}")
print(f"RNN Mean Absolute Error: {rnn_mae}")

# Visualize the predicted values from RNN
plt.figure(figsize=(12, 6))
plt.scatter(y_test.index, y_test.values, label='Actual', color='red')
plt.scatter(y_test.index, y_pred_rnn, label='RNN Predicted', alpha=0.5, color='blue')
plt.title('Actual vs RNN Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Train an SVM model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(X_train, y_train)

# Make predictions using the SVM model
y_pred_svm = svm_model.predict(X_test)

# Model evaluation for SVM
svm_mse = mean_squared_error(y_test, y_pred_svm)
svm_mae = mean_absolute_error(y_test, y_pred_svm)
print(f"SVM Mean Squared Error: {svm_mse}")
print(f"SVM Mean Absolute Error: {svm_mae}")

# Visualize the predicted values from SVM
plt.figure(figsize=(12, 6))
plt.scatter(y_test.index, y_test.values, label='Actual', color='red')
plt.scatter(y_test.index, y_pred_svm, label='SVM Predicted', alpha=0.5, color='purple')
plt.title('Actual vs SVM Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Visualize all the predictions (Linear Regression, RNN, SVM)
plt.figure(figsize=(12, 6))
plt.scatter(y_test.index, y_test.values, label='Actual', color='red')
plt.scatter(y_test.index, y_pred, label='Linear Regression Predicted', alpha=0.5, color='green')
plt.scatter(y_test.index, y_pred_rnn, label='RNN Predicted', alpha=0.5, color='blue')
plt.scatter(y_test.index, y_pred_svm, label='SVM Predicted', alpha=0.5, color='purple')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
