import sqlite3
import matplotlib.pyplot as plt
import tensorflow as tf
import config


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file path
    :return: Connection object or None
    """
    conn = sqlite3.connect(db_file)
    return conn


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img, config.IMG_WIDTH, config.IMG_HEIGHT)
    return img, label


def decode_img(img, width, height):
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    img = (tf.cast(img, tf.float32) / 127.5) - 1
    return tf.image.resize(img, [width, height])

