import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

podaci = pd.read_csv('occupancy_processed.csv')
X = podaci[['S3_Temp', 'S5_CO2']].values
y = podaci['Room_Occupancy_Count'].astype(int)

X_trening, X_test, y_trening, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True, random_state=123
)

skaliranje = StandardScaler()
X_trening_sk = skaliranje.fit_transform(X_trening)
X_test_sk = skaliranje.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_trening_sk, y_trening)

y_pred = knn.predict(X_test_sk)

matrica_zabune = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(matrica_zabune, display_labels=["Slobodna", "Zauzeta"]).plot(cmap="Oranges")
plt.title("Matrica zabune - KNN klasifikator")
plt.grid(False)
plt.show()

print("Rezultati KNN klasifikatora:\n")
print(classification_report(y_test, y_pred, target_names=["Slobodna", "Zauzeta"]))
