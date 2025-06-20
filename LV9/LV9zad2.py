import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
import numpy as np
import os

train_folder = 'Train'
test_folder = 'Test'

train_dataset = image_dataset_from_directory(
    train_folder,
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(48, 48),
    batch_size=32
)

validation_dataset = image_dataset_from_directory(
    train_folder,
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(48, 48),
    batch_size=32
)

test_dataset = image_dataset_from_directory(
    test_folder,
    labels='inferred',
    label_mode='categorical',
    image_size=(48, 48),
    batch_size=32
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

cnn_model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(48, 48, 3)),
    layers.Conv2D(32, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(43, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
tensorboard_callback = TensorBoard(log_dir='logs')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback]
)

test_loss, test_accuracy = cnn_model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy:.4f}')

true_labels = np.concatenate([labels for images, labels in test_dataset], axis=0)
true_labels = np.argmax(true_labels, axis=1)

predicted_probs = cnn_model.predict(test_dataset)
predicted_labels = np.argmax(predicted_probs, axis=1)

conf_mat = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:\n", conf_mat)
