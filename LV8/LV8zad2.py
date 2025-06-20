import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage.color import rgb2gray
from tensorflow.keras.models import load_model
import numpy as np

putanja_slike = 'test.png'

# Učitaj i pripremi sliku
slika_org = mpimg.imread(putanja_slike)
slika_gray = rgb2gray(slika_org)
slika_resize = resize(slika_gray, (28, 28))

# Prikaži sliku
plt.imshow(slika_resize, cmap='gray')
plt.axis('off')
plt.show()

# Priprema za mrežu
slika_input = slika_resize.reshape(1, 28, 28, 1).astype('float32')

# Učitaj prethodno istrenirani model
model = load_model('best_model.h5')

# Napravi predikciju
rezultat = model.predict(slika_input)
klasa_pred = np.argmax(rezultat)

# Prikaz rezultata
print(f'Predviđena klasa: {klasa_pred}')
