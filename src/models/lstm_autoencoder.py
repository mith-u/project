import numpy as np, joblib
from tensorflow import keras
from tensorflow.keras import layers

X = np.random.rand(100, 10, 1)  # dummy sequence data
model = keras.Sequential([
    layers.LSTM(16, activation="relu", input_shape=(10,1), return_sequences=True),
    layers.LSTM(8, activation="relu", return_sequences=False),
    layers.RepeatVector(10),
    layers.LSTM(8, activation="relu", return_sequences=True),
    layers.LSTM(16, activation="relu", return_sequences=True),
    layers.TimeDistributed(layers.Dense(1))
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, X, epochs=1, batch_size=16)

model.save("artifacts/models/lstm_ae.keras")

print("Saved -> artifacts/models/lstm_ae.h5")
