import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, max_error, mean_absolute_error

#Učitavanje podataka
df = pd.read_csv('cars_processed.csv')
print(df.info())

#Odabir značajki i ciljne varijable
znacajke = df[['km_driven', 'year', 'engine', 'max_power']]
cilj = df['selling_price']

#Podjela na skupove za učenje i testiranje
X_train, X_test, y_train, y_test = train_test_split(znacajke, cilj, test_size=0.2, random_state=300)

#Standardizacija značajki
skaliraj = StandardScaler()
X_train_s = skaliraj.fit_transform(X_train)
X_test_s = skaliraj.transform(X_test)

#Treniranje modela
model = LinearRegression()
model.fit(X_train_s, y_train)

#Predikcija
y_pred_train = model.predict(X_train_s)
y_pred_test = model.predict(X_test_s)

#Evaluacija
print("R2 (test):", r2_score(y_test, y_pred_test))
print("RMSE (test):", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Maksimalna pogreška (test):", max_error(y_test, y_pred_test))
print("MAE (test):", mean_absolute_error(y_test, y_pred_test))

#Vizualizacija rezultata
plt.figure(figsize=(12, 8))
graf = sns.regplot(x=y_pred_test, y=y_test, line_kws={'color': 'green'})
graf.set(xlabel='Predviđena cijena', ylabel='Stvarna cijena', title='Usporedba stvarne i predviđene cijene')
plt.show()
