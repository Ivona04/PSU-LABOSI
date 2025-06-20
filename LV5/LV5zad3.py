import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

podatci = pd.read_csv('occupancy_processed.csv')

X = podatci[['S3_Temp', 'S5_CO2']].values
y = podatci['Room_Occupancy_Count'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

skaliraj = StandardScaler()
X_train = skaliraj.fit_transform(X_train)
X_test = skaliraj.transform(X_test)

stablo_model = DecisionTreeClassifier(random_state=42)
stablo_model.fit(X_train, y_train)

y_pred = stablo_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Slobodna", "Zauzeta"]).plot(cmap="Oranges")
plt.title("Matrica zabune - Stablo odlučivanja")
plt.grid(False)
plt.show()

print("Rezultati klasifikatora:\n")
print(classification_report(y_test, y_pred, target_names=["Slobodna", "Zauzeta"]))

plt.figure(figsize=(12, 6))
plot_tree(stablo_model, feature_names=["S3_Temp", "S5_CO2"], class_names=["Slobodna", "Zauzeta"], filled=True)
plt.title("Vizualizacija stabla odlučivanja")
plt.show()
