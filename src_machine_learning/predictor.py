import os

import tensorflow as tf
import helpers
import numpy as np
import config
import matplotlib as plt


def get_models(model_name):
    directory = os.path.join(os.path.dirname(os.getcwd()), "generated\\")

    default_model = tf.keras.Sequential(
        [tf.keras.models.load_model(directory + r"{0}\{0}_default_model.h5".format(model_name)),
         tf.keras.layers.Softmax()])

    optimized_model = tf.keras.Sequential(
        [tf.keras.models.load_model(directory + r"{0}\{0}_optimized_model.h5".format(model_name)),
         tf.keras.layers.Softmax()])

    pretrained_model = tf.keras.Sequential(
        [tf.keras.models.load_model(directory + r"{0}\{0}_pretrained_model.h5".format(model_name)),
         tf.keras.layers.Softmax()])

    return default_model, optimized_model, pretrained_model


def predict_age(img):
    pass


def predict_color(img):
    pass


def predict_engine(img):
    pass


def predict_doors(img):
    pass


def get_all_predictions(img_path):
    img = get_image_from_path(img_path)

    predictions = {"body_shape": predict_body_shape(img),
                   "age": predict_age(img),
                   "color": predict_color(img),
                   "engine": predict_engine(img),
                   "doors": predict_doors(img)}
    return predictions


def predict_body_shape(img):
    predictions = get_model_predictions(img, body_shape_default_model, body_shape_optimized_model,
                                        body_shape_pretrained_model, config.id_to_body_shape)
    return predictions


def get_model_predictions(img, default_model, optimized_model, pretrained_model, mapping_dict):
    default_prediction = default_model.predict(img)[0].tolist()
    optimized_prediction = optimized_model.predict(img)[0].tolist()
    pretrained_prediction = pretrained_model.predict(img)[0].tolist()

    predictions_dict = {"default": [], "optimized": [], "pretrained": []}
    for i in range(1, len(default_prediction)):
        predictions_dict["default"].append((mapping_dict[i], default_prediction[i]))
        predictions_dict["optimized"].append((mapping_dict[i], optimized_prediction[i]))
        predictions_dict["pretrained"].append((mapping_dict[i], pretrained_prediction[i]))
    return predictions_dict


def get_image_from_path(img_path):
    img = helpers.preprocess_image(img_path, "")[0]
    img = np.expand_dims(img, axis=0)
    return img


body_shape_default_model, body_shape_optimized_model, body_shape_pretrained_model = get_models("body_shape")
# age_default_model, age_optimized_model, age_pretrained_model = get_models("age")
# color_default_model, color_optimized_model, color_pretrained_model = get_models("color")
# doors_default_model, doors_optimized_model, doors_pretrained_model = get_models("doors")
# engine_default_model, engine_optimized_model, engine_pretrained_model = get_models("engine")
