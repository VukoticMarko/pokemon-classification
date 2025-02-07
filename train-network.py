import cv2 as cv
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'pokemon-dataset'

full_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    seed=123,
    image_size=(128, 128),
    shuffle=True,
    batch_size=32
)

dataset_size = len(full_ds)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size) # 10% val
test_size = dataset_size - train_size - val_size # Test is 100% - 80% - 10% = rest of the dataset

train_ds = full_ds.take(train_size)
remaining = full_ds.skip(train_size)
val_ds = remaining.take(val_size)
test_ds = remaining.skip(val_size)

def noise(image):
    '''
    Function that adds random noise to the image
    '''
    noise_tensor = tf.random.normal(shape=tf.shape(image)[1:], mean=0.0, stddev=0.05)
    noise_tensor = tf.expand_dims(noise_tensor, axis=0)
    return tf.clip_by_value(image + noise_tensor, 0.0, 1.0)

def data_augmentation(image, label):

    '''
    Augmentation function that normalizes the images and then
    randomly adds preprocessing to the image based on the
    percentages below
    '''

    image = image / 255.0
    if tf.random.uniform(()) < 0.15: # 15% chance to flip horizontally
        image = tf.image.flip_left_right(image)
    if tf.random.uniform(()) < 0.15: # 15% chance to flip vertically
        image = tf.image.flip_up_down(image)
    if tf.random.uniform(()) < 0.15: # 15% chance to increase brightness
        image = tf.image.adjust_brightness(image, delta=0.2)
    if tf.random.uniform(()) < 0.15: # 15% chance to add contrast change
        image = tf.image.adjust_contrast(image, 0.8)
    if tf.random.uniform(()) < 0.1: # 10% chance to rotate image by 90 deg
        image = tf.image.rot90(image)
    if tf.random.uniform(()) < 0.1: # 10% chance to grayscale image
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image) # RGB format but without color info
    if tf.random.uniform(()) < 0.1: # 10% chance to call noise function
        image = noise(image)
    if tf.random.uniform(()) < 0.1: # 10% chance to adjust hue
        image = tf.image.random_hue(image, max_delta=0.08) 
    if tf.random.uniform(()) < 0.1: # 10% chance to crop/resize
        def crop_resize(img):
            cropped = tf.image.random_crop(img, size=[112, 112, 3])
            return tf.image.resize(cropped, [128, 128])
        image = tf.map_fn(crop_resize, image)
    if tf.random.uniform(()) < 0.1: # 10% chance to simulate zooming
        def zoom(img):
            scale = tf.random.uniform(shape=[], minval=0.8, maxval=1.2)
            new_h = tf.cast(tf.cast(tf.shape(img)[0], tf.float32) * scale, tf.int32)
            new_w = tf.cast(tf.cast(tf.shape(img)[1], tf.float32) * scale, tf.int32)
            resized = tf.image.resize(img[tf.newaxis, ...], (new_h, new_w))[0]
            return tf.image.resize_with_crop_or_pad(resized, 128, 128)
        image = tf.map_fn(zoom, image)

    return image, label

def preprocess(image, label):
    '''
    Function used for preprocessing validation and test datasets
    '''
    return image / 255.0, label  

train_ds = train_ds.map(data_augmentation).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# Added checkpointing to retreive best epoch from the model
# and save it afterwards
checkpoint = ModelCheckpoint(
    "model/weights/best_weights/best_pokemon_weights.keras",
    monitor="val_accuracy", 
    save_best_only=True,
    mode="max",
    verbose=1
)

# The Model
model = keras.models.Sequential([
    Conv2D(12, kernel_size=3, activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=3, activation="relu"),
    Dropout(0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=3, activation="relu"),
    Dropout(0.2),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(18, kernel_size=3, activation="relu"),
    Dropout(0.1),
    Conv2D(24, kernel_size=3, activation="relu"),
    Conv2D(32, kernel_size=3, activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=3, activation="relu", padding="same"),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(20, activation="relu"),
    Dense(100, activation="relu"),
    Dense(1000, activation="softmax")
])

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=1000, callbacks=[checkpoint])

test_loss, test_acc = model.evaluate(test_ds)


# Plot model accuracy and loss on datasets
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Save the model
model.save("model/pokemon-model.keras")

# Serialize model to JSON file
model_json = model.to_json()
with open("model/pokemon-model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
model.save_weights("model/weights/pokemon.weights.h5")

# Serialize weights to Keras
model.save("model/weights/pokemon.weights.keras")

# Evaluate the model on test dataset and save Metrics to txt file
test_loss, test_acc = model.evaluate(test_ds)
with open("model/metrics.txt", "w") as f:
    f.write(f"------------------------------")
    f.write(f"Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Train Loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"------------------------------")
    f.write(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
    f.write(f"Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
    f.write(f"------------------------------")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")

print('------------------------------')
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print('------------------------------')