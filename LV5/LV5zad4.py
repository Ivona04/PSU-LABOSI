from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

podatci = pd.read_csv("occupancy_processed.csv")

X = podatci[['S3_Temp', 'S5_CO2']].values
y = podatci['Room_Occupancy_Count'].values

X_trening, X_test, y_trening, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

skaliraj = StandardScaler()
X_trening = skaliraj.fit_transform(X_trening)
X_test = skaliraj.transform(X_test)

model_log = LogisticRegression()
model_log.fit(X_trening, y_trening)

y_pred = model_log.predict(X_test)

matrica_zabune = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(matrica_zabune, display_labels=["Slobodna", "Zauzeta"]).plot(cmap="Oranges")
plt.title("Matrica zabune - Logistička regresija")
plt.grid(False)
plt.show()

print("Točnost modela:", accuracy_score(y_test, y_pred))
print("Izvještaj klasifikacije:\n")
print(classification_report(y_test, y_pred, target_names=['Slobodna', 'Zauzeta']))
