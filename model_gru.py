# =========================================================
# model_gru.py
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

# =========================================================
# CONFIGURATION MACROS 
# =========================================================

# ---- feature switches ----
USE_FUTURE_INCL       = True        # makes it more volatile, which is visually better for the predicted curve, almost same MAE
USE_RELATIVE_INCL     = False       # not really relevant
INCLUDE_PREV_TORQUE   = False       # usually False for GRU

# ---- temporal parameters ----
SEQ_LEN      = 1
WINDOW_INCL  = 10
DT           = 1.0

# ---- fatigue / integral parameters ----
MM = 100.0        # meters (m)
PP = 400.0        # power threshold (W)

# ---- training ----
GRU_UNITS   = 32
EPOCHS      = 100
BATCH_SIZE = 32

# ---- validation / test windows ----
VAL_START  = 250
VAL_END    = 325
TEST_START = 325
TEST_END   = 425

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_pickle("data/dataset.pkl")

time  = df["time"].values
speed = df["speed"].values
cadence_pedal = df["cadence_pedal"].values
inclination_rad_tan_smooth = df["inclination"].values
power = df["power"].values
torque_wheel = df["torque_wheel"].values

# =========================================================
# PRECOMPUTE INTEGRALS 
# =========================================================

N = len(power)

energy_spent = np.zeros(N)
time_above_pp = np.zeros(N) # time with power > Power threshold (400W)

for t in range(1, N):
    energy_spent[t] = energy_spent[t - 1] + power[t] * DT
    time_above_pp[t] = time_above_pp[t - 1] + (power[t] > PP) * DT

# =========================================================
# FEATURE BUILDER 
# =========================================================

def build_features_gru(t, torque_prev):
    features = [
        cadence_pedal[t],
        speed[t],
        energy_spent[t],
        time_above_pp[t],
    ]

    if INCLUDE_PREV_TORQUE:
        features.append(torque_prev)

    if USE_FUTURE_INCL:
        incl_window = inclination_rad_tan_smooth[t:t + WINDOW_INCL]
        if USE_RELATIVE_INCL:
            incl_window = incl_window - incl_window[0]
        features.extend(incl_window)
    else:
        features.append(inclination_rad_tan_smooth[t])

    return np.asarray(features)

# =========================================================
# AUTOREGRESSIVE GRU VALIDATION
# =========================================================

def autoregressive_validation_gru(model, scaler):
    y_preds, y_true = [], []

    seq = []
    t0 = TEST_START

    for k in range(t0 - SEQ_LEN, t0):
        torque_prev = torque_wheel[k - 1]
        seq.append(build_features_gru(k, torque_prev))

    seq = np.asarray(seq)  # (SEQ_LEN, n_features)

    # ---- autoregressive rollout ----
    for t in range(TEST_START, TEST_END):
        x = scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(1, SEQ_LEN, -1)

        y_hat = model.predict(x, verbose=0)[0, 0]

        y_preds.append(y_hat)
        y_true.append(torque_wheel[t])

        # build next feature using prediction
        next_feat = build_features_gru(t, y_hat)
        seq = np.vstack([seq[1:], next_feat])

    return np.array(y_true), np.array(y_preds)

# =========================================================
# BUILD GRU SEQUENCES
# =========================================================

X_seq, y_seq = [], []

for t in range(SEQ_LEN, N - WINDOW_INCL):
    seq = []
    for k in range(t - SEQ_LEN, t):
        torque_prev = torque_wheel[k - 1]
        seq.append(build_features_gru(k, torque_prev))

    X_seq.append(seq)
    y_seq.append(torque_wheel[t])

X_seq = np.asarray(X_seq)   # (samples, SEQ_LEN, features)
y_seq = np.asarray(y_seq)

# =========================================================
# TRAIN / VAL / TEST SPLIT
# =========================================================

X_train = np.concatenate((X_seq[:VAL_START], X_seq[TEST_END:]))
X_val   = X_seq[VAL_START:VAL_END]
X_test  = X_seq[TEST_START:TEST_END]

y_train = np.concatenate((y_seq[:VAL_START], y_seq[TEST_END:]))
y_val   = y_seq[VAL_START:VAL_END]
y_test  = y_seq[TEST_START:TEST_END]

valid_idx = y_train > 0
X_train = X_train[valid_idx]
y_train = y_train[valid_idx]

# =========================================================
# SCALING (FLATTEN TIME AXIS)
# =========================================================

scaler = StandardScaler()

ns, ts, nf = X_train.shape

X_train = scaler.fit_transform(
    X_train.reshape(-1, nf)
).reshape(ns, ts, nf)

X_val = scaler.transform(
    X_val.reshape(-1, nf)
).reshape(X_val.shape)

X_test = scaler.transform(
    X_test.reshape(-1, nf)
).reshape(X_test.shape)

# =========================================================
# MODEL
# =========================================================

model = Sequential([
    GRU(GRU_UNITS, input_shape=(SEQ_LEN, nf)),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(1e-3),
    loss="mse",
    metrics=["mae"]
)

early_stop = callbacks.EarlyStopping(
    monitor="val_mae",
    patience=100,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# =========================================================
# EVALUATION
# =========================================================

y_test_true, y_test_pred = autoregressive_validation_gru(model, scaler)

print("GRU (autoregressive) MAE :", mean_absolute_error(y_test_true, y_test_pred))
print("GRU (autoregressive) RMSE:", np.sqrt(mean_squared_error(y_test_true, y_test_pred)))

# =========================================================
# PLOTS Scatter and Time Series
# =========================================================

plt.figure(figsize=(12, 6))
plt.scatter(y_test_true, y_test_pred, alpha=0.5)
plt.plot(
    [min(y_test_true), max(y_test_true)],
    [min(y_test_true), max(y_test_true)],
    linestyle="--"
)
plt.xlabel("Actual Torque (Nm)")
plt.ylabel("Predicted Torque (Nm)")
plt.title("GRU (autoregressive): Actual vs Predicted Torque")
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(
    time[TEST_START:TEST_END],
    y_test_true,
    label="Actual Torque"
)
plt.plot(
    time[TEST_START:TEST_END],
    y_test_pred,
    label="Predicted Torque",
    linestyle="--"
)
plt.xlabel("Time (seconds)")
plt.ylabel("Torque (Nm)")
plt.title("Actual vs Predicted Torque over Time | Best GRU Model")
plt.legend()
plt.grid()
plt.show()


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

INPUT_FILE  = "../Implementation/TimeTorqueInclinationSpeed_rad_tan.txt"
OUTPUT_FILE = "../Implementation/TimeTorqueInclinationSpeed_rad_tanPrediction.txt"


# ---------------------------------------------------------
# SLICE TEST RANGE & APPEND PREDICTION COLUMN
# ---------------------------------------------------------

data = np.loadtxt(INPUT_FILE, skiprows=2)

data_test = data[TEST_START:TEST_END]

if len(data_test) != len(y_test_pred):
    raise ValueError("Mismatch between data rows and predictions")

data_with_pred = np.column_stack([data_test, y_test_pred])

# ---------------------------------------------------------
# WRITE NEW FILE WITH UPDATED HEADER
# ---------------------------------------------------------

n_rows, n_cols = data_with_pred.shape

with open(OUTPUT_FILE, "w") as f:
    f.write("#1\n")
    f.write(
        f"double TimeTorqueInclinationSpeed_rad_tanPrediction"
        f"({n_rows},{n_cols})\n"
    )
    np.savetxt(
        f,
        data_with_pred,
        fmt="%.6f",
        delimiter="\t"
    )

print(f"Saved: {OUTPUT_FILE}")
