import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

def kreiraj_podatke(broj_uzoraka, tip):
    if tip == 1:
        X, y = datasets.make_blobs(n_samples=broj_uzoraka, random_state=365)
    elif tip == 2:
        X, y = make_blobs(n_samples=broj_uzoraka, random_state=148)
        transformacija = [[0.6083, -0.6366], [-0.4088, 0.8525]]
        X = np.dot(X, transformacija)
    elif tip == 3:
        X, y = make_blobs(n_samples=broj_uzoraka, centers=4,
                          cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=148)
    elif tip == 4:
        X, y = datasets.make_circles(n_samples=broj_uzoraka, factor=0.5, noise=0.05)
    elif tip == 5:
        X, y = datasets.make_moons(n_samples=broj_uzoraka, noise=0.05)
    else:
        X, y = np.empty((0, 2)), []
    return X

uzorci = 500
metoda = 5

podaci = kreiraj_podatke(uzorci, metoda)

Z = linkage(podaci, method='ward')

plt.figure(figsize=(9, 6))
dendrogram(Z)
plt.title('Dendrogram - Hijerarhijsko klasteriranje (Ward)')
plt.xlabel('Indeksi uzoraka')
plt.ylabel('Udaljenost')
plt.tight_layout()
plt.show()
