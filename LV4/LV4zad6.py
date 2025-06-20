import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

# Učitavanje podataka
podaci = pd.read_csv('cars_processed.csv')
print(podaci.info())

# Kodiranje kategorijskih podataka
podaci = pd.get_dummies(podaci, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

# Definiranje značajki i ciljne varijable
X = podaci.drop(columns=['name', 'selling_price', 'seats'])
y = podaci['selling_price']

# Podjela na trening i test skup
X_trening, X_test, y_trening, y_test = train_test_split(X, y, test_size=0.2, random_state=300)

# Standardizacija podataka
skaler = StandardScaler()
X_trening_std = skaler.fit_transform(X_trening)
X_test_std = skaler.transform(X_test)

# Kreiranje i treniranje modela
model_lin = LinearRegression()
model_lin.fit(X_trening_std, y_trening)

# Predikcija
y_pred_trening = model_lin.predict(X_trening_std)
y_pred_test = model_lin.predict(X_test_std)

# Evaluacija modela
print("R2 na test setu:", r2_score(y_test, y_pred_test))
print("RMSE na test setu:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Maksimalna pogreška na test setu:", max_error(y_test, y_pred_test))
print("MAE na test setu:", mean_absolute_error(y_test, y_pred_test))

# Vizualizacija rezultata
plt.figure(figsize=(13, 10))
graf = sns.regplot(x=y_pred_test, y=y_test, line_kws={'color': 'green'})
graf.set(xlabel='Predviđena cijena', ylabel='Stvarna cijena', title='Rezultati na testnim podacima')
plt.show()
