from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

slika = mpimg.imread('example_grayscale.png')
if slika.ndim == 3:
    slika = np.mean(slika, axis=2)

pikseli = slika.reshape(-1, 1)

# broj klastera za kvantizaciju
k = 10
kmeans_model = cluster.KMeans(n_clusters=k, n_init=10)
kmeans_model.fit(pikseli)

# zamjena piksela sa centroid vrijednostima
centri = kmeans_model.cluster_centers_.squeeze()
oznake = kmeans_model.labels_
kompresirana = np.choose(oznake, centri).reshape(slika.shape)

plt.figure()
plt.imshow(slika, cmap='gray')
plt.title('Originalna slika')
plt.axis('off')

plt.figure()
plt.imshow(kompresirana, cmap='gray')
plt.title(f'Kvantizirana slika (k = {k})')
plt.axis('off')
plt.show()
