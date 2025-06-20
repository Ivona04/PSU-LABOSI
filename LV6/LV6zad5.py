import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread, imsave

img = imread("example.png")
h, w, c = img.shape

pixels = img.reshape(-1, 3)

#kvantizacija sa 12 boja (klastera)
num_clusters = 12
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(pixels)

pixels_quantized = kmeans.cluster_centers_[kmeans.labels_]

img_quantized = pixels_quantized.reshape(h, w, c).astype(np.uint8)

imsave("quantized_example.png", img_quantized)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Originalna slika")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_quantized)
plt.title("Kvantizirana slika")
plt.axis('off')

plt.tight_layout()
plt.show()
