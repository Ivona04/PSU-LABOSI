import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

url = 'http://iszz.azo.hr/iskzl/rs/podatak/export/xml?postaja=160&polutant=5&tipPodatka=0&vrijemeOd=01.01.2017&vrijemeDo=31.12.2017'

#Učitavanje podataka sa URL-a
podaci_xml = urllib.request.urlopen(url).read()

#Parsiranje XML podataka u DataFrame
df = pd.read_xml(podaci_xml)

#Izbor relevantnih stupaca i njihovo preimenovanje
df = df[['vrijednost', 'vrijeme']].rename(columns={'vrijednost':'koncentracija', 'vrijeme':'datum'})

#Pretvaranje tipova podataka
df['koncentracija'] = df['koncentracija'].astype(float)
df['datum'] = pd.to_datetime(df['datum'], utc=True)

#Tri najveće koncentracije PM10 u 2017.
najvece_tri = df.nlargest(3, 'koncentracija')

print("Tri najviša PM10 mjerenja u Osijeku tijekom 2017.:")
print(najvece_tri[['datum', 'koncentracija']])

#Crtanje linijskog grafa koncentracije kroz godinu
plt.figure(figsize=(10, 5))
plt.plot(df['datum'], df['koncentracija'], label='PM10 (µg/m³)')
plt.title('PM10 koncentracije zraka u Osijeku (2017)')
plt.xlabel('Datum')
plt.ylabel('Koncentracija PM10 (µg/m³)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
