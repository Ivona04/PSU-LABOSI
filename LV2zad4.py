import numpy as np
import matplotlib.pyplot as plt

def napravi_sahovnicu(velicina_polja, broj_redova, broj_stupaca):
    crno_polje = np.zeros((velicina_polja, velicina_polja), dtype=np.uint8)
    bijelo_polje = np.ones((velicina_polja, velicina_polja), dtype=np.uint8) * 255

    red_1 = np.hstack([crno_polje, bijelo_polje] * (broj_stupaca // 2))
    red_2 = np.hstack([bijelo_polje, crno_polje] * (broj_stupaca // 2))

    ploca = np.vstack([red_1, red_2] * (broj_redova // 2))

    return ploca

velicina = 100
redovi = 6
stupci = 6

sahovnica = napravi_sahovnicu(velicina, redovi, stupci)

plt.imshow(sahovnica, cmap='gray', vmin=0, vmax=255)
plt.axis('off')  #ukloni osi za ljepši prikaz
plt.show()
