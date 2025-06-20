import numpy as np
import matplotlib.pyplot as plt

#Učitavanje slike
slika = plt.imread("tiger.png")

#Povećavamo svjetlinu slike, ali pazimo da vrijednosti ostanu u opsegu [0,1]
pojacana_svjetlina = np.clip(slika * 1.5, 0, 1)

#Okrećemo sliku za 90 stupnjeva u smjeru kazaljke na satu
okrenuta_slika = np.rot90(slika, k=3)  # isto kao k=-1

#Horizontalno zrcalimo sliku
horizontalno_zrcalo = np.fliplr(slika)

#Smanjujemo rezoluciju slike tako da uzimamo svaki 10-ti piksel
korak = 10
smanjena_rezolucija = slika[::korak, ::korak]

#Kreiramo praznu (crnu) sliku istih dimenzija
prazna_slika = np.zeros_like(slika)

#Kopiramo drugu četvrtinu slike (širine) u praznu sliku
visina, sirina, _ = slika.shape
prazna_slika[:, sirina//4 : sirina//2] = slika[:, sirina//4 : sirina//2]

#Pripremamo prikaz u jednom redu, 5 podgrafova
figura, axes = plt.subplots(1, 5, figsize=(15, 5))

axes[0].imshow(pojacana_svjetlina)
axes[0].set_title("Povećana svjetlina")

axes[1].imshow(okrenuta_slika)
axes[1].set_title("Rotacija")

axes[2].imshow(horizontalno_zrcalo)
axes[2].set_title("Zrcalo")

axes[3].imshow(smanjena_rezolucija)
axes[3].set_title("Smanjena rezolucija")

axes[4].imshow(prazna
