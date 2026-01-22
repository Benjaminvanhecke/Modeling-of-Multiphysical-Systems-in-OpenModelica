# =========================================================
# data_processing.py
# =========================================================

import numpy as np
import pandas as pd
import csv

# =========================================================
# PARAMETERS
# =========================================================

D_wheel = 0.7
eta = 0.90 # drivetrain efficiency 
g = 9.81
vwind_cte = 4

# =========================================================
# LOAD RAW CSV 
# =========================================================

with open("velo_data.csv", "r") as file:
    reader = csv.reader(file)
    data = list(reader)

first_15_cols = [row[:15] for row in data]
data = np.array(first_15_cols[1:], dtype=float)

# =========================================================
# EXTRACT SIGNALS 
# =========================================================

time = data[:, 0] - data[:, 0][0]
distance = data[:, 3]
speed = np.where(data[:, 5] == 0, 0.1, data[:, 5])
altitude = data[:, 6]
power = data[:, 7]
cadence_pedal = np.where(
    data[:, 10] == 0, 0.1, data[:, 10] / 60 * 2 * np.pi
)

# =========================================================
# INCLINATION 
# =========================================================

dalt = np.diff(altitude, prepend=altitude[0])
ddist = np.diff(distance, prepend=distance[0])

inclination = np.arctan2(dalt, ddist) * 180 / np.pi
inclination[0] = 0

# Wherever ddist == 0, replace with previous inclination
same_distance_idx = (ddist == 0)

for i in np.where(same_distance_idx)[0]:
    if i > 0:
        inclination[i] = inclination[i-1]

# Filter inclination to remove extreme values (artifacts)
for i in range(1, len(inclination)):
    if abs(inclination[i] - inclination[i - 1]) > 15:
        inclination[i] = inclination[i - 1]

inclination_rad_tan = np.tan(inclination * np.pi / 180) # alt / dist

# =========================================================
# TORQUES 
# =========================================================

radius_wheel = D_wheel / 2
torque_pedal = power / cadence_pedal
torque_wheel = power * eta * radius_wheel / speed

# =========================================================
# SAVE EVERYTHING AS PANDAS
# =========================================================

df = pd.DataFrame({
    "time": time,
    "speed": speed,
    "cadence_pedal": cadence_pedal,
    "inclination": inclination_rad_tan,
    "power": power,
    "torque_wheel": torque_wheel
})

df.to_pickle("data/dataset.pkl")

# =========================================================
# EXPORT TIME–TORQUE–INCLINATION–SPEED 
# =========================================================

TimeTorqueInclinationSpeed = np.column_stack((time, torque_wheel, inclination_rad_tan,speed))

n_rows, n_cols = TimeTorqueInclinationSpeed.shape

with open("../Implementation/TimeTorqueInclinationSpeed_rad_tan.txt","w") as f:
    f.write("#1\n")
    f.write(f"double TimeTorqueInclinationSpeed({n_rows},{n_cols})\n")
    np.savetxt(
        f,
        TimeTorqueInclinationSpeed,
        fmt="%.6f",
        delimiter="\t"
    )