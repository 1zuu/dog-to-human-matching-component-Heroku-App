import os
import requests
from flask import Flask
from flask import jsonify
from flask import request
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from heroku_inference import InferenceModel
model = InferenceModel()

from variables import *
from util import *

app = Flask(__name__)

# message = {
#     "text" : "I would like to arrange a playdate for my female small size Maltese puppy it is very playful,active and have a good behaviour with other pets and behave well with strangers love go for walks. we live in kalutara.",
#     "label" : "shih tzu",
#     "feature" : "['flat', 'no', 6, 'femmale', 'adult', 'medium']"
#     }

@app.route("/predict", methods=["GET", "POST"])
def predict(show_fig=False):
    message = request.get_json(force=True)
    if len(message) == 3:
        text_pad, feature, label = get_prediction_data(message)
        model.extract_image_features(label)
        n_neighbours = model.predictions(text_pad, feature, show_fig)
        response = {
            "neighbours": n_neighbours
                    }
        return jsonify(response)
    else:
        return "Please input both Breed and the text content"

if __name__ == "__main__": 
    app.run(debug=True, host='0.0.0.0', port= 5000, threaded=False, use_reloader=False)
