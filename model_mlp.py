# =========================================================
# model_mlp.py
# =========================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_pickle("data/dataset.pkl")

time = df["time"].values
speed = df["speed"].values
cadence_pedal = df["cadence_pedal"].values
inclination_rad_tan_smooth = df["inclination"].values
torque_wheel = df["torque_wheel"].values

# =========================================================
# ORIGINAL CONFIG FLAGS 
# =========================================================

# ---- feature switches ----
USE_FUTURE_INCL = False           # makes it more volatile, good in combination with prev torque false
WINDOW_INCL = 10
USE_RELATIVE_INCL = False         # not very useful
INCLUDE_PREV_TORQUE = False       # makes it smoother, not always better

# ---- validation / test windows ----
VAL_START = 250
VAL_END = 325
TEST_START = 325
TEST_END = 425

# =========================================================
# FEATURE BUILDER 
# =========================================================

def build_features(t, cadence_pedal, speed, torque_prev, inclination):
    base_features = [
        cadence_pedal[t],
        speed[t],
    ]

    if INCLUDE_PREV_TORQUE:
        base_features.append(torque_prev)

    if USE_FUTURE_INCL:
        incl_window = inclination[t:t + WINDOW_INCL]
        if USE_RELATIVE_INCL:
            incl_window = incl_window - incl_window[0]
    else:
        incl_window = [inclination[t]]

    return np.concatenate([base_features, incl_window])

# =========================================================
# DATASET BUILD 
# =========================================================

X, y = [], []

for t in range(1, len(torque_wheel) - WINDOW_INCL):
    X.append(
        build_features(
            t,
            cadence_pedal,
            speed,
            torque_wheel[t - 1],
            inclination_rad_tan_smooth
        )
    )
    y.append(torque_wheel[t])

X = np.asarray(X)
y = np.asarray(y)

X_train = np.concatenate((X[:VAL_START], X[TEST_END:]))
X_val = X[VAL_START:TEST_END]
X_test = X[TEST_START:TEST_END]

y_train = np.concatenate((y[:VAL_START], y[TEST_END:]))
y_val = y[VAL_START:TEST_END]
y_test = y[TEST_START:TEST_END]

valid = y_train > 0
X_train = X_train[valid]
y_train = y_train[valid]

# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =========================================================
# MODEL 
# =========================================================

model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(0.001),
    loss="mse",
    metrics=["mae"]
)

early_stop = callbacks.EarlyStopping(
    monitor="val_mae",
    patience=100,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =========================================================
# AUTOREGRESSIVE TEST 
# =========================================================

def autoregressive_validation():
    y_preds, y_true = [], []
    y_prev = torque_wheel[TEST_START - 1]

    for t in range(TEST_START, TEST_END):
        x = build_features(
            t, cadence_pedal, speed, y_prev, inclination_rad_tan_smooth
        ).reshape(1, -1)

        x = scaler.transform(x)
        y_hat = model.predict(x, verbose=0)[0, 0]

        y_preds.append(y_hat)
        y_true.append(torque_wheel[t])
        y_prev = y_hat

    return np.array(y_true), np.array(y_preds)

y_true, y_pred = autoregressive_validation()

print("MLP MAE :", mean_absolute_error(y_true, y_pred))
print("MLP RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

# =========================================================
# PLOTS 
# =========================================================

import matplotlib.pyplot as plt

y_test_true = y_true
y_test_pred = y_pred

# =========================================================
# Scatter plot: Actual vs Predicted
# =========================================================

plt.figure(figsize=(12, 6))
plt.scatter(y_test_true, y_test_pred, alpha=0.5)
plt.plot(
    [min(y_test_true), max(y_test_true)],
    [min(y_test_true), max(y_test_true)],
    linestyle='--'
)
plt.xlabel('Actual Torque (Nm)')
plt.ylabel('Predicted Torque (Nm)')
plt.title('Actual vs Predicted Torque')
plt.grid()
plt.show()

# =========================================================
# Time series plot
# =========================================================

plt.figure(figsize=(12, 6))
plt.plot(
    time[TEST_START:TEST_END],
    y_test_true,
    label='Actual Torque'
)
plt.plot(
    time[TEST_START:TEST_END],
    y_test_pred,
    label='Predicted Torque',
    linestyle='--'
)
plt.xlabel('Time (seconds)')
plt.ylabel('Torque (Nm)')
plt.title('Actual vs Predicted Torque over Time | Best MLP Model')
plt.legend()
plt.grid()
plt.show()

