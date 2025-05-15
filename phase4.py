import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
n_sensors = 4         # Number of sensors on the structure
n_readings = 500      # Number of data points (time series length)

# Time index for simulation (e.g., hourly readings)
time_index = pd.date_range(start="2025-01-01", periods=n_readings, freq='H')

# Simulate sensor readings (strain, temperature, vibration)
def generate_sensor_data(mean, std):
    return np.random.normal(loc=mean, scale=std, size=n_readings)

data = {
    f"strain_sensor_{i+1}": generate_sensor_data(100, 5) for i in range(n_sensors)
}
data.update({
    f"temp_sensor_{i+1}": generate_sensor_data(25, 1.5) for i in range(n_sensors)
})
data.update({
    f"vibration_sensor_{i+1}": generate_sensor_data(0.02, 0.005) for i in range(n_sensors)
})

# Create DataFrame for sensor data
df = pd.DataFrame(data, index=time_index)

# Define anomaly thresholds
strain_threshold = 120
temp_threshold = 40
vibration_threshold = 0.05

# Anomaly detection function
def detect_anomalies(data, threshold):
    return data > threshold

# Anomaly detection for each sensor type
strain_anomalies = df.filter(like="strain").apply(detect_anomalies, threshold=strain_threshold)
temp_anomalies = df.filter(like="temp").apply(detect_anomalies, threshold=temp_threshold)
vibration_anomalies = df.filter(like="vibration").apply(detect_anomalies, threshold=vibration_threshold)

# Combine all anomalies
anomalies = strain_anomalies | temp_anomalies | vibration_anomalies

# Mark anomaly rows in the original data
df['anomaly'] = anomalies.any(axis=1)

# Visualization of anomalies
plt.figure(figsize=(10, 6))
plt.title("Structural Health Monitoring - Anomaly Detection")
plt.plot(df.index, df['strain_sensor_1'], label="Strain Sensor 1")
plt.plot(df.index, df['temp_sensor_1'], label="Temperature Sensor 1")
plt.plot(df.index, df['vibration_sensor_1'], label="Vibration Sensor 1")
plt.scatter(df.index[df['anomaly']], df['strain_sensor_1'][df['anomaly']], color='red', label="Anomaly", marker='x')
plt.xlabel("Time")
plt.ylabel("Sensor Readings")
plt.legend()
plt.show()

# Save the processed data and anomaly report
df.to_csv("structural_health_monitoring.csv")
print("Data saved to structural_health_monitoring.csv")
