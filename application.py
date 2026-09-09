from functools import lru_cache
from pathlib import Path
import math
import pickle
from flask import Flask, render_template, request

application = Flask(__name__)
app = application
FIELDS = ('Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region')


@lru_cache(maxsize=1)
def load_models():
    """Load the repository's trusted model artifacts independently of cwd."""
    directory = Path(__file__).resolve().parent / 'models'
    with (directory / 'ridge.pkl').open('rb') as model_file:
        model = pickle.load(model_file)
    with (directory / 'scaler.pkl').open('rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    try:
        values = [float(request.form[field]) for field in FIELDS]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Non-finite input')
    except (KeyError, TypeError, ValueError):
        return {'error': 'Provide a finite numeric value for every input field.', 'fields': list(FIELDS)}, 400
    model, scaler = load_models()
    prediction = model.predict(scaler.transform([values]))
    return render_template('home.html', results=prediction[0])


if __name__ == '__main__':
    app.run()
