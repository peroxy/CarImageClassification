import tensorflow as tf
import os
import numpy as np
import helpers
import matplotlib.pyplot as plt


def get_default_model(img_size, mapping_dict):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(img_size, img_size, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(max(mapping_dict.values()) + 1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def get_optimized_model(img_size, mapping_dict):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 3)),
        tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(120, activation='softmax'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(max(mapping_dict.values()) + 1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def get_pretrained_model(img_size, mapping_dict):
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False)
    mobile_net.trainable = False
    model = tf.keras.models.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(max(mapping_dict.values()) + 1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model(model, rows, mapping_dict, save_model_path, batch_size, epochs):
    train_number = int(len(rows) * 0.8)
    train_filenames = [row[1] for row in rows][:train_number]
    train_labels = [mapping_dict[row[0]] for row in rows][:train_number]
    train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels)))
    train_data = (train_data.map(helpers.preprocess_image).shuffle(buffer_size=train_number).repeat().batch(batch_size))

    test_number = int(len(rows) * 0.2)
    test_filenames = [row[1] for row in rows][-test_number:]
    test_labels = [mapping_dict[row[0]] for row in rows][-int(len(rows) * 0.2):]
    test_data = tf.data.Dataset.from_tensor_slices((tf.constant(test_filenames), tf.constant(test_labels)))
    test_data = (test_data.map(helpers.preprocess_image).batch(batch_size))

    steps_per_epoch = train_number // batch_size
    validation_steps = test_number // batch_size
    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=test_data,
                        validation_steps=validation_steps,
                        steps_per_epoch=steps_per_epoch)

    model.summary()
    model.save(save_model_path)
    return model, history


def draw_model_history(history, epochs, file_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(file_name, bbox_inches='tight')
