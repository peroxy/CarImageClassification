from flask import Flask
from predictor import get_all_predictions

app = Flask(__name__)


@app.route('/predict/<path:img_path>')
def predict(img_path):
    predictions = get_all_predictions(img_path)
    return predictions
