# Structural Health Monitoring (SHM) System

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

# Simulate sensor data acquisition (strain gauge and accelerometer)
def acquire_sensor_data():
    strain = random.uniform(0.8, 1.2)  # Simulated strain gauge reading
    acceleration = random.uniform(-0.5, 0.5)  # Simulated accelerometer reading
    return strain, acceleration

# Anomaly detection function
def detect_anomaly(strain, acceleration):
    if strain < 0.9 or strain > 1.1 or abs(acceleration) > 0.4:
        return True
    return False

# Real-time data monitoring and visualization
def monitor_structure(duration=10):
    data = {'Time': [], 'Strain': [], 'Acceleration': [], 'Anomaly': []}
    start_time = time.time()
    while time.time() - start_time < duration:
        strain, acceleration = acquire_sensor_data()
        anomaly = detect_anomaly(strain, acceleration)
        timestamp = time.time() - start_time

        # Save data
        data['Time'].append(timestamp)
        data['Strain'].append(strain)
        data['Acceleration'].append(acceleration)
        data['Anomaly'].append(anomaly)

        # Print real-time status
        status = 'Anomaly Detected!' if anomaly else 'Normal'
        print(f"Time: {timestamp:.2f}s | Strain: {strain:.2f} | Acceleration: {acceleration:.2f} | Status: {status}")

        time.sleep(0.5)

    # Data visualization
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time'], df['Strain'], label='Strain')
    plt.plot(df['Time'], df['Acceleration'], label='Acceleration')
    plt.scatter(df['Time'], df['Anomaly'], color='red', label='Anomaly')
    plt.xlabel('Time (s)')
    plt.ylabel('Sensor Readings')
    plt.title('Real-Time Structural Health Monitoring')
    plt.legend()
    plt.show()

# Run the monitoring system
monitor_structure(10)
