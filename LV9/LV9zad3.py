import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Učitaj prethodno istrenirani model
model = tf.keras.models.load_model('best_model.h5')

# Putanja do slike za predikciju
slika_put = 'znak.jpeg'

# Učitaj i prilagodi sliku
slika = image.load_img(slika_put, target_size=(48, 48))
slika_niz = image.img_to_array(slika) / 255.0
slika_niz = np.expand_dims(slika_niz, axis=0)

# Predikcija klase
predikcija = model.predict(slika_niz)
klasa_pred = np.argmax(predikcija)

print(f'Predviđena klasa: {klasa_pred}')

# Prikaz slike s naslovom predikcije
plt.imshow(slika)
plt.title(f'Predviđena klasa: {klasa_pred}')
plt.axis('off')
plt.show()
