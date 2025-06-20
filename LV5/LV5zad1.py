import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = 'occupancy_processed.csv'
data = pd.read_csv(data_path)
print(f"Ukupno primjera u datasetu: {len(data)}")

features = ['S3_Temp', 'S5_CO2']
label = 'Room_Occupancy_Count'
class_labels = {0: 'Slobodno', 1: 'Zauzeto'}

X = data[features].values
y = data[label].values

plt.figure(figsize=(8, 6))
for class_val in np.unique(y):
    plt.scatter(
        X[y == class_val, 0],
        X[y == class_val, 1],
        label=class_labels[class_val],
        alpha=0.7
    )

plt.xlabel('Temperatura (S3_Temp)')
plt.ylabel('Razina CO2 (S5_CO2)')
plt.title('Dijagram raspršenja zauzetosti prostorije')
plt.legend()
plt.show()
