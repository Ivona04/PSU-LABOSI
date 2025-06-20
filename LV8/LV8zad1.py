import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

#Učitavanje MNIST podataka i normalizacija
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

#One-hot enkodiranje labela
train_labels_cat = to_categorical(train_labels, 10)
test_labels_cat = to_categorical(test_labels, 10)

def create_cnn_model():
    cnn_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name="conv1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Conv2D(64, (3, 3), activation='relu', name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation='relu', name="dense1"),
        layers.Dense(10, activation='softmax', name="output")
    ])
    return cnn_model

model = create_cnn_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = os.path.join("logs_tensorboard")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_callback = callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy')

model.fit(
    train_images, train_labels_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[tensorboard_callback, checkpoint_callback]
)

best_model = tf.keras.models.load_model("best_model.h5")

train_loss, train_accuracy = best_model.evaluate(train_images, train_labels_cat, verbose=0)
test_loss, test_accuracy = best_model.evaluate(test_images, test_labels_cat, verbose=0)

print(f"Točnost na trening skupu: {train_accuracy:.4f}")
print(f"Točnost na testnom skupu: {test_accuracy:.4f}")

train_pred_labels = np.argmax(best_model.predict(train_images), axis=1)
test_pred_labels = np.argmax(best_model.predict(test_images), axis=1)

train_cm = confusion_matrix(train_labels, train_pred_labels)
test_cm = confusion_matrix(test_labels, test_pred_labels)

print("\nMatrica zabune za trening skup:")
ConfusionMatrixDisplay(confusion_matrix=train_cm).plot(cmap="Blues")
plt.show()

print("\nMatrica zabune za testni skup:")
ConfusionMatrixDisplay(confusion_matrix=test_cm).plot(cmap="Blues")
plt.show()
