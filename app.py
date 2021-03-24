from __future__ import division, print_function
# coding=utf-8
import os

# Keras
from keras.models import model_from_json

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from model_files.ml_model import make_prediction, get_labels

# Define a flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')

## Model reconstruction
WEIGHTS_PATH = 'model_files/cnn_dogvision_weights.h5'
ARCH_PATH = 'model_files/model.json'

# load json and create model
json_file = open(ARCH_PATH, 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn_model = model_from_json(loaded_model_json)
# load weights into new model
cnn_model.load_weights(WEIGHTS_PATH)
# load models for feature extraction
from keras.applications.inception_resnet_v2 import InceptionResNetV2
InceptionResNetV2(input_shape=(331,331,3), include_top=False, weights='imagenet')
from keras.applications.xception import Xception
Xception(input_shape=(331,331,3), include_top=False, weights='imagenet')
from keras.applications.inception_v3 import InceptionV3
InceptionV3(input_shape=(331,331,3), include_top=False, weights='imagenet')
from keras.applications.nasnet import NASNetLarge
NASNetLarge(input_shape=(331,331,3), include_top=False, weights='imagenet')
print("Model successfully loaded")


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = make_prediction(cnn_model, file_path)

        # Process your result for human
        result = get_labels(preds)  # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)