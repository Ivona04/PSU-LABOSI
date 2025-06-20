import numpy as np
import pandas as pd

podaci = pd.read_csv('mtcars.csv')

# Prvih 5 automobila s najvećom potrošnjom (najmanji mpg)
top5_najveća_potrosnja = podaci.sort_values(by='mpg').head(5)
print(top5_najveća_potrosnja)

# Automobili s 8 cilindara
automobili_8cil = podaci[podaci['cyl'] == 8]

# 3 automobila s najmanjom potrošnjom među 8 cilindara
najmanja_potrosnja_3 = automobili_8cil.sort_values(by='mpg').head(3)
print(najmanja_potrosnja_3)

# Automobili sa 6 cilindara
automobili_6cil = podaci[podaci['cyl'] == 6]

# Prosječna potrošnja za 6 cilindara
prosjek_6cil = automobili_6cil['mpg'].mean()

# Automobili s 4 cilindra i težinom između 2.0 i 2.2
automobili_4cil_tezak = podaci[(podaci['cyl'] == 4) & (podaci['wt'] >= 2.0) & (podaci['wt'] <= 2.2)]

# Prosječna potrošnja za gore navedene automobile
prosjek_4cil_tezak = automobili_4cil_tezak['mpg'].mean()
print(prosjek_4cil_tezak)

# Broj automobila po tipu mjenjača
broj_mjenjaca = podaci['am'].value_counts()
print("0 znači automatski mjenjač, 1 znači ručni mjenjač")
print(broj_mjenjaca)

# Broj automobila s automatskim mjenjačem i preko 100 KS
broj_hp_preko100 = podaci[(podaci['am'] == 0) & (podaci['hp'] > 100)].shape[0]
print(broj_hp_preko100)

# Dodavanje stupca s masom u kilogramima (wt je u tisućama funti)
podaci['masa_kg'] = podaci['wt'] * 1000 * 0.453592
print(podaci[['masa_kg']])
