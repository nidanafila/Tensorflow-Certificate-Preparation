# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
import shutil

def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    TRAINING_DIR = "data/rps/"
    rock = os.path.join(TRAINING_DIR, 'rock')
    paper = os.path.join(TRAINING_DIR, 'paper')
    scissor = os.path.join(TRAINING_DIR, 'scissors')
    path_train = os.path.join(TRAINING_DIR, 'train')
    path_test = os.path.join(TRAINING_DIR, 'test')

    if os.path.isdir(path_train) == False:
        os.mkdir(path_train)
    if os.path.isdir(path_test) == False:
        os.mkdir(path_test)

    if os.path.isdir(os.path.join(path_train, 'rock')) == False:
        os.mkdir(os.path.join(path_train, 'rock'))
    for i in os.listdir(rock)[:700]:
        path_file = os.path.join(rock, i)
        shutil.copy(path_file, os.path.join(path_train, 'rock'))
    if os.path.isdir(os.path.join(path_test, 'rock')) == False:
        os.mkdir(os.path.join(path_test, 'rock'))
    for i in os.listdir(rock)[700:]:
        path_file = os.path.join(rock, i)
        shutil.copy(path_file, os.path.join(path_test, 'rock'))

    if os.path.isdir(os.path.join(path_train, 'paper')) == False:
        os.mkdir(os.path.join(path_train, 'paper'))
    for i in os.listdir(paper)[:700]:
        path_file = os.path.join(paper, i)
        shutil.copy(path_file, os.path.join(path_train, 'paper'))
    if os.path.isdir(os.path.join(path_test, 'paper')) == False:
        os.mkdir(os.path.join(path_test, 'paper'))
    for i in os.listdir(paper)[700:]:
        path_file = os.path.join(paper, i)
        shutil.copy(path_file, os.path.join(path_test, 'paper'))

    if os.path.isdir(os.path.join(path_train, 'scissor')) == False:
        os.mkdir(os.path.join(path_train, 'scissor'))
    for i in os.listdir(scissor)[:700]:
        path_file = os.path.join(scissor, i)
        shutil.copy(path_file, os.path.join(path_train, 'scissor'))
    if os.path.isdir(os.path.join(path_test, 'scissor')) == False:
        os.mkdir(os.path.join(path_test, 'scissor'))
    for i in os.listdir(scissor)[700:]:
        path_file = os.path.join(scissor, i)
        shutil.copy(path_file, os.path.join(path_test, 'scissor'))

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    train_generator = train_datagen.flow_from_directory(
        path_train,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=35
    )

    validation_generator = test_datagen.flow_from_directory(
        path_test,
        target_size=(150, 150),
        class_mode='categorical',
        batch_size=12
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=50,
        steps_per_epoch=20,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=20,
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save("model_B3.h5")
