import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Step 1: Simulate Sensor Data
np.random.seed(42)
n_samples = 1000
normal_data = np.random.normal(loc=0, scale=1, size=(n_samples, 3))  # Simulating normal sensor readings

# Inject anomalies (simulate structural damage)
n_anomalies = 50
anomalies = np.random.normal(loc=5, scale=1.5, size=(n_anomalies, 3))
data = np.vstack([normal_data, anomalies])

# Convert to DataFrame
columns = ['Vibration', 'Strain', 'Displacement']
df = pd.DataFrame(data, columns=columns)

# Step 2: Normalize Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Anomaly Detection using Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = model.fit_predict(scaled_data)
df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})  # 1 = anomaly

# Step 4: Visualization
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['Vibration'], c=df['Anomaly'], cmap='coolwarm', label='Anomaly')
plt.title('Structural Health Monitoring: Vibration Readings')
plt.xlabel('Sample Index')
plt.ylabel('Vibration')
plt.legend()
plt.grid(True)
plt.show()
