import os
import sys

import config
import helpers
from model import train_model, draw_model_history, get_default_model, get_optimized_model, get_pretrained_model


def train_body_shape_models(limit, project_dir):
    connection = helpers.create_connection(config.database_path)
    with connection:
        cur = connection.cursor()
        cur.execute(r"SELECT BodyShape, LocalPicturePath FROM Cars "
                    r"WHERE LocalPicturePath IS NOT NULL AND "
                    r"LocalPicturePath != '' AND "
                    r"BodyShape != '' AND "
                    r"BodyShape IS NOT NULL "
                    r"LIMIT {}".format(limit))
        body_shapes = cur.fetchall()

        model, history = train_model(get_default_model(config.IMG_SIZE, config.body_shape_to_id),
                                     body_shapes,
                                     config.body_shape_to_id,
                                     project_dir + r"\generated\body_shape\body_shape_default_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\body_shape\body_shape_default_model_history.png")

        model, history = train_model(get_pretrained_model(config.IMG_SIZE, config.body_shape_to_id),
                                     body_shapes,
                                     config.body_shape_to_id,
                                     project_dir + r"\generated\body_shape\body_shape_pretrained_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\body_shape\body_shape_pretrained_model_history.png")

        model, history = train_model(get_optimized_model(config.IMG_SIZE, config.body_shape_to_id),
                                     body_shapes,
                                     config.body_shape_to_id,
                                     project_dir + r"\generated\body_shape\body_shape_optimized_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\body_shape\body_shape_optimized_model_history.png")


def train_color_models(limit, project_dir):
    connection = helpers.create_connection(config.database_path)
    with connection:
        cur = connection.cursor()
        cur.execute(
            r"SELECT Color, LocalPicturePath FROM Cars "
            r"WHERE LocalPicturePath IS NOT NULL AND "
            r"LocalPicturePath != '' AND "
            r"Color != '' AND "
            r"Color IS NOT NULL "
            r"LIMIT {}".format(limit))
        colors = cur.fetchall()

        model, history = train_model(get_default_model(config.IMG_SIZE, config.color_to_id),
                                     colors,
                                     config.color_to_id,
                                     project_dir + r"\generated\color\color_default_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\color\color_default_model_history.png")

        model, history = train_model(get_pretrained_model(config.IMG_SIZE, config.color_to_id),
                                     colors,
                                     config.color_to_id,
                                     project_dir + r"\generated\color\color_pretrained_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\color\color_pretrained_model_history.png")

        model, history = train_model(get_optimized_model(config.IMG_SIZE, config.color_to_id),
                                     colors,
                                     config.color_to_id,
                                     project_dir + r"\generated\color\color_optimized_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\color\color_optimized_model_history.png")


def train_age_models(limit, project_dir):
    connection = helpers.create_connection(config.database_path)
    with connection:
        cur = connection.cursor()
        cur.execute(
            r"SELECT Age, LocalPicturePath FROM Cars "
            r"WHERE LocalPicturePath IS NOT NULL AND "
            r"LocalPicturePath != '' AND "
            r"Age IS NOT NULL "
            r"LIMIT {}".format(limit))
        ages = cur.fetchall()

        model, history = train_model(get_default_model(config.IMG_SIZE, config.age_to_id),
                                     ages,
                                     config.age_to_id,
                                     project_dir + r"\generated\age\age_default_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\age\age_default_model_history.png")

        model, history = train_model(get_pretrained_model(config.IMG_SIZE, config.age_to_id),
                                     ages,
                                     config.age_to_id,
                                     project_dir + r"\generated\age\age_pretrained_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\age\age_pretrained_model_history.png")

        model, history = train_model(get_optimized_model(config.IMG_SIZE, config.age_to_id),
                                     ages,
                                     config.age_to_id,
                                     project_dir + r"\generated\age\age_optimized_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\age\age_optimized_model_history.png")


def train_doors_models(limit, project_dir):
    connection = helpers.create_connection(config.database_path)
    with connection:
        cur = connection.cursor()
        cur.execute(
            r"SELECT DoorNumber, LocalPicturePath FROM Cars "
            r"WHERE LocalPicturePath IS NOT NULL AND "
            r"LocalPicturePath != '' AND "
            r"DoorNumber != '' AND "
            r"DoorNumber IS NOT NULL "
            r"LIMIT {}".format(limit))
        doors = cur.fetchall()

        model, history = train_model(get_default_model(config.IMG_SIZE, config.door_to_id),
                                     doors,
                                     config.door_to_id,
                                     project_dir + r"\generated\doors\doors_default_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\doors\doors_default_model_history.png")

        model, history = train_model(get_pretrained_model(config.IMG_SIZE, config.door_to_id),
                                     doors,
                                     config.door_to_id,
                                     project_dir + r"\generated\doors\doors_pretrained_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS,
                           project_dir + r"\generated\doors\doors_pretrained_model_history.png")

        model, history = train_model(get_optimized_model(config.IMG_SIZE, config.door_to_id),
                                     doors,
                                     config.door_to_id,
                                     project_dir + r"\generated\doors\doors_optimized_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS,
                           project_dir + r"\generated\doors\doors_optimized_model_history.png")


def train_engine_models(limit, project_dir):
    connection = helpers.create_connection(config.database_path)
    with connection:
        cur = connection.cursor()
        cur.execute(
            r"SELECT EngineType, LocalPicturePath FROM Cars "
            r"WHERE LocalPicturePath IS NOT NULL AND "
            r"LocalPicturePath != '' AND "
            r"EngineType != '' AND "
            r"EngineType IS NOT NULL "
            r"LIMIT {}".format(limit))
        engines = cur.fetchall()

        model, history = train_model(get_default_model(config.IMG_SIZE, config.engine_to_id),
                                     engines,
                                     config.engine_to_id,
                                     project_dir + r"\generated\engine\engine_default_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\engine\engine_default_model_history.png")

        model, history = train_model(get_pretrained_model(config.IMG_SIZE, config.engine_to_id),
                                     engines,
                                     config.engine_to_id,
                                     project_dir + r"\generated\engine\engine_pretrained_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\engine\engine_pretrained_model_history.png")

        model, history = train_model(get_optimized_model(config.IMG_SIZE, config.engine_to_id),
                                     engines,
                                     config.engine_to_id,
                                     project_dir + r"\generated\engine\engine_optimized_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, project_dir + r"\generated\engine\engine_optimized_model_history.png")


# we have around 45k train cases, we want them all..
directory = os.path.dirname(os.getcwd())
max_row_count = 1000
train_body_shape_models(max_row_count, directory)
train_color_models(max_row_count, directory)
train_age_models(max_row_count, directory)
train_engine_models(max_row_count, directory)
train_doors_models(max_row_count, directory)
