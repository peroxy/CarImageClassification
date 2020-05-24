import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import helpers
import config

connection = helpers.create_connection(config.database_path)

with connection:
    cur = connection.cursor()

    cur.execute(r"SELECT BodyShape, LocalPicturePath FROM Cars WHERE LocalPicturePath like '%\train\%'")
    trainRows = cur.fetchall()

    train_filenames = [row[1] for row in trainRows][:1000]
    train_labels = [config.bodyShapes[row[0]] for row in trainRows][:1000]
    train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels)))
    train_data = (train_data.map(helpers.preprocess_image).shuffle(buffer_size=10000).batch(config.BATCH_SIZE))

    cur.execute(r"SELECT BodyShape, LocalPicturePath FROM Cars WHERE LocalPicturePath like '%\test\%'")
    testRows = cur.fetchall()

    test_filenames = [row[1] for row in testRows][:100]
    test_labels = [config.bodyShapes[row[0]] for row in testRows][:100]
    test_data = tf.data.Dataset.from_tensor_slices((tf.constant(test_filenames), tf.constant(test_labels)))
    test_data = (test_data.map(helpers.preprocess_image).shuffle(buffer_size=10000).batch(config.BATCH_SIZE))

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    neki = helpers.preprocess_image(r"F:\NoBackupHere\AvtoNetPictures\train\637255073426611273.jpg", "")
    steps_per_epoch = 1000 // config.BATCH_SIZE
    validation_steps = 100 // config.BATCH_SIZE
    model.fit(train_data.repeat(), epochs=5, validation_data=test_data.repeat(), validation_steps=validation_steps,
              steps_per_epoch=steps_per_epoch)
    model.summary()

    # model.predict(neki[0], verbose=1)
    # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # predictions = probability_model.predict(test_images)

# TODO: save model https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format
