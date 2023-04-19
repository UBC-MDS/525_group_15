from flask import Flask, request, jsonify
import joblib
import numpy as np
from io import BytesIO
import boto3
import joblib

## Import any other packages that are needed

app = Flask(__name__)

# 1. Load your model here
#modify this to use the s3 model.joblib
s3 = boto3.resource('s3')
bucket_str = "mds-525-group-15"
bucket_key = "output/model.joblib"
with BytesIO() as data:
    s3.Bucket(bucket_str).download_fileobj(bucket_key, data)
    data.seek(0)    # move back to the beginning after writing
    model = joblib.load(data)

# 2. Define a prediction function
def return_prediction(content):

    # format input_data here so that you can pass it to model.predict()
    test_data = [content['data']]
    return model.predict(test_data)

# 3. Set up home page using basic html
@app.route("/")
def index():
    # feel free to customize this if you like
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 25 climate model outputs.
    """

# 4. define a new route which will accept POST requests and return model predictions
@app.route('/predict', methods=['POST'])
def rainfall_prediction():
    content = request.json  # this extracts the JSON content we sent
    prediction = return_prediction(content)[0]
    results = {'Input: ':content['data'] ,
               'Prediction: ': prediction}  # return whatever data you wish, it can be just the prediction
                     # or it can be the prediction plus the input data, it's up to you
    return jsonify(results)
