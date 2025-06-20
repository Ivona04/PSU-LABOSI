import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cars_processed.csv')
print(df.info())

#Najjeftiniji i najskuplji automobil
df_sorted = df.sort_values("selling_price")
print(df_sorted.iloc[[0, -1]])

#Broj vozila iz 2012.
broj_2012 = (df["year"] == 2012).sum()
print("Broj vozila proizvedenih 2012.:", broj_2012)

#Prosječna kilometraža za benzin i dizel
km_prosjek = df.groupby("fuel")["km_driven"].mean()
print("Prosječna kilometraža (Petrol):", km_prosjek.get("Petrol", 0))
print("Prosječna kilometraža (Diesel):", km_prosjek.get("Diesel", 0))

#Grafički prikazi
sns.pairplot(df, hue='fuel')
sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')

#Uklanjanje kolona koje ne koristimo
df = df.drop(['name', 'mileage'], axis=1)

#Brojači za objektne kolone
objekt_kolone = df.select_dtypes(include=['object']).columns.tolist()
fig, axes = plt.subplots(1, len(objekt_kolone), figsize=(15, 5))
for col, ax in zip(objekt_kolone, axes):
    sns.countplot(x=df[col], ax=ax)

#Boxplot i histogram
df.boxplot(by='fuel', column=['selling_price'], grid=False)
df['selling_price'].hist(grid=False)

plt.show()
