import tensorflow as tf
import helpers
import numpy as np
import config

img = helpers.preprocess_image(r"F:\NoBackupHere\AvtoNetPictures\test\637255014345349195.jpg", "")[0]
img = np.expand_dims(img, axis=0)

# model = tf.keras.models.load_model("bodyShape_model.h5")
# predictions = model.predict(img, verbose=1)
# print("{} : {}".format(np.argmax(predictions[0]), config.id_to_body_shape[np.argmax(predictions[0])]))

model = tf.keras.models.load_model("color_model.h5")
predictions = model.predict(img, verbose=0)
print("{} : {}".format(np.argmax(predictions[0]), config.id_to_color[np.argmax(predictions[0])]))

model = tf.keras.models.load_model("age_model.h5")
predictions = model.predict(img, verbose=0)
print("{} : {}".format(np.argmax(predictions[0]), config.id_to_age[np.argmax(predictions[0])]))

