import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Učitavanje podataka MNIST
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

#Prikaz nekoliko primjera iz trening skupa
for i in range(5, 10):
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Broj: {train_labels[i]}")
    plt.axis('off')
    plt.show()

#Normalizacija podataka
train_images_scaled = train_images.astype("float32") / 255
test_images_scaled = test_images.astype("float32") / 255

#Reshape u vektore duljine 784
train_images_scaled = train_images_scaled.reshape(-1, 784)
test_images_scaled = test_images_scaled.reshape(-1, 784)

#One-hot enkodiranje klasa
train_labels_cat = keras.utils.to_categorical(train_labels, 10)
test_labels_cat = keras.utils.to_categorical(test_labels, 10)

#Definicija modela
model = Sequential([
    keras.Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Treniranje modela
history = model.fit(train_images_scaled, train_labels_cat, epochs=5, batch_size=32)

#Evaluacija na trening i test skupu
train_loss, train_acc = model.evaluate(train_images_scaled, train_labels_cat, verbose=0)
test_loss, test_acc = model.evaluate(test_images_scaled, test_labels_cat, verbose=0)

print(f"Točnost na trening setu: {train_acc:.2f}")
print(f"Točnost na testnom setu: {test_acc:.2f}")

#Predikcije na testnom skupu
test_preds = model.predict(test_images_scaled)
test_pred_classes = np.argmax(test_preds, axis=1)

#Matrica zabune za testni skup
cm_test = confusion_matrix(test_labels, test_pred_classes)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=range(10))
disp_test.plot()
plt.title("Matrica zabune - testni skup")
plt.show()

#Predikcije na trening skupu
train_preds = model.predict(train_images_scaled)
train_pred_classes = np.argmax(train_preds, axis=1)

#Matrica zabune za trening skup
cm_train = confusion_matrix(train_labels, train_pred_classes)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=range(10))
disp_train.plot()
plt.title("Matrica zabune - trening skup")
plt.show()

#Prikaz prvih 5 netočnih predikcija na testnom skupu
netocni_indeksi = np.where(test_labels != test_pred_classes)[0]
for i in range(5):
    idx = netocni_indeksi[i]
    plt.imshow(test_images[idx], cmap='gray')
    plt.title(f"Stvarno: {test_labels[idx]}, Predviđeno: {test_pred_classes[idx]}")
    plt.axis('off')
    plt.show()
