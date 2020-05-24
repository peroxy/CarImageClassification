import config
import helpers
from model import train_model, draw_model_history, get_default_model


def train_body_shape_model(limit):
    connection = helpers.create_connection(config.database_path)
    with connection:
        cur = connection.cursor()
        cur.execute(r"SELECT BodyShape, LocalPicturePath FROM Cars "
                    r"WHERE LocalPicturePath LIKE '%\train\%' AND "
                    r"BodyShape != '' AND "
                    r"BodyShape is not null "
                    r"LIMIT {}}".format(limit))
        body_shapes = cur.fetchall()

        model, history = train_model(get_default_model(config.IMG_SIZE, config.body_shape_to_id),
                                     body_shapes,
                                     config.body_shape_to_id,
                                     r"generated\bodyShape_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, r"generated\body_shape_history.png")


def train_color_model(limit):
    connection = helpers.create_connection(config.database_path)
    with connection:
        cur = connection.cursor()
        cur.execute(
            r"SELECT Color, LocalPicturePath FROM Cars "
            r"WHERE LocalPicturePath LIKE '%\train\%' AND "
            r"Color != '' AND "
            r"Color IS NOT NULL "
            r"LIMIT {}}".format(limit))
        colors = cur.fetchall()

        model, history = train_model(get_default_model(config.IMG_SIZE, config.color_to_id),
                                     colors,
                                     config.color_to_id,
                                     r"generated\color_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, r"generated\color_model_history.png")


def train_age_model(limit):
    connection = helpers.create_connection(config.database_path)
    with connection:
        cur = connection.cursor()
        cur.execute(
            r"SELECT Age, LocalPicturePath FROM Cars "
            r"WHERE LocalPicturePath LIKE '%\train\%' AND "
            r"Age != '' AND "
            r"Age IS NOT NULL "
            r"LIMIT {}}".format(limit))
        ages = cur.fetchall()

        model, history = train_model(get_default_model(config.IMG_SIZE, config.age_to_id),
                                     ages,
                                     config.age_to_id,
                                     r"generated\age_model.h5",
                                     config.BATCH_SIZE,
                                     config.EPOCHS)

        draw_model_history(history, config.EPOCHS, r"generated\age_model_history.png")


# we have like max 28k train cases, we want max
max_row_count = 50000
train_body_shape_model(max_row_count)
train_color_model(max_row_count)
train_age_model(max_row_count)
