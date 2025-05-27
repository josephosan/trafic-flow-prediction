import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load the dataset
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# 2. Preprocess date and feature engineering
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek

# Drop unnecessary columns
df = df.drop(columns=['holiday', 'weather_description', 'date_time'])

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['weather_main'], drop_first=True)

# 3. Normalize features
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)

# 4. Sequence generation (past 24 hours to predict next hour)
def create_sequences(data, window_size=24):
    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(data[i-window_size:i])
        y.append(data[i, df.columns.get_loc('traffic_volume')])
    return np.array(x), np.array(y)

window_size = 24
X, y = create_sequences(scaled_df, window_size)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 6. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 7. Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# 8. Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")

# 9. Plot predictions
preds = model.predict(X_test)

plt.figure(figsize=(14, 6))
plt.plot(y_test[:300], label='True')
plt.plot(preds[:300], label='Predicted')
plt.legend()
plt.title("Traffic Volume Prediction - True vs Predicted")
plt.show()