import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import make_blobs

def kreiraj_podatke(velicina, tip):
    if tip == 1:
        rnd = 365
        X, y = datasets.make_blobs(n_samples=velicina, random_state=rnd)
    elif tip == 2:
        rnd = 148
        X, y = make_blobs(n_samples=velicina, random_state=rnd)
        transform = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transform)
    elif tip == 3:
        rnd = 148
        X, y = make_blobs(n_samples=velicina, centers=4,
                          cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=rnd)
    elif tip == 4:
        X, y = datasets.make_circles(n_samples=velicina, factor=0.5, noise=0.05)
    elif tip == 5:
        X, y = datasets.make_moons(n_samples=velicina, noise=0.05)
    else:
        X, y = np.empty((0, 2)), []
    return X, y

def izracunaj_inertiju(podaci, raspon_k):
    inertije = []
    for k in raspon_k:
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        model.fit(podaci)
        inertije.append(model.inertia_)
    return inertije

velicina_skupa = 500
tip_podataka = 5

X, y = kreiraj_podatke(velicina_skupa, tip_podataka)

kriteriji = range(2, 21)
inertija = izracunaj_inertiju(X, kriteriji)

plt.figure(figsize=(8,5))
plt.plot(kriteriji, inertija, marker='o', linestyle='-')
plt.title('Elbow metoda za određivanje k')
plt.xlabel('Broj klastera (k)')
plt.ylabel('Inercija (vrijednost kriterijske funkcije)')
plt.grid(True)
plt.tight_layout()
plt.show()
