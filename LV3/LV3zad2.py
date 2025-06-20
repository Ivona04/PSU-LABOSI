import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

podaci = pd.read_csv('mtcars.csv')

#Izračun prosječne potrošnje po broju cilindara
prosjek_mpg = podaci.groupby('cyl')['mpg'].mean()

plt.figure(figsize=(8, 4))
plt.bar(prosjek_mpg.index, prosjek_mpg, color=['navy', 'forestgreen', 'crimson'])
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna potrošnja (mpg)')
plt.title('Prosječna potrošnja po cilindrima')
plt.show()

#Boxplot težine automobila po broju cilindara
plt.figure(figsize=(8, 4))
podaci.boxplot(column='wt', by='cyl', grid=False)
plt.xlabel('Broj cilindara')
plt.ylabel('Težina (1000 lbs)')
plt.title('Raspodjela težine automobila')
plt.show()

#Boxplot potrošnje prema tipu mjenjača
plt.figure(figsize=(8, 4))
podaci.boxplot(column='mpg', by='am', grid=False)
plt.xlabel('Tip mjenjača (0=Automatski, 1=Ručni)')
plt.ylabel('Potrošnja (mpg)')
plt.title('Potrošnja po tipu mjenjača')
plt.show()

#Scatter plot snage vs ubrzanja, boje po tipu mjenjača
plt.figure(figsize=(8, 5))
boje = ['red' if t == 0 else 'blue' for t in podaci['am']]
plt.scatter(podaci['hp'], podaci['qsec'], c=boje)
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (qsec)')
plt.title('Ubrzanje vs snaga po tipu mjenjača')
plt.legend(['Automatski', 'Ručni'], loc='upper right')
plt.show()
