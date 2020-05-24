import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import helpers
import config


def preprocess_data(rows):
    data = []
    counter = 0
    for row in rows:
        image = helpers.preprocess_image(row[1], config.IMG_HEIGHT, config.IMG_WIDTH)
        data.append((image.numpy(), config.bodyShapes[row[0]]))
        counter += 1
        print("Pre-processing progress: " + '{0:.2f}'.format(100 * (counter / len(rows))) + "%")
    return data


connection = helpers.create_connection(config.database_path)

with connection:
    cur = connection.cursor()

    cur.execute(r"SELECT BodyShape, LocalPicturePath FROM Cars WHERE LocalPicturePath like '%\train\%'")
    trainRows = cur.fetchall()
    train_data = preprocess_data(trainRows[:1000])

    cur.execute(r"SELECT BodyShape, LocalPicturePath FROM Cars WHERE LocalPicturePath like '%\test\%'")
    testRows = cur.fetchall()
    test_data = preprocess_data(testRows[:500])

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# TODO: save model https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format
