from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import numpy as np
import pandas as pd
import logging
import json_log_formatter
from ddtrace import patch_all, tracer

# Apply automatic patching for supported libraries
patch_all()

app = Flask(__name__)
swagger = Swagger(app)

# Load the trained model
model = joblib.load('model_churn.pkl')

# Configure JSON logging
formatter = json_log_formatter.JSONFormatter()
json_handler = logging.FileHandler(filename='/var/log/my-log.json')
json_handler.setFormatter(formatter)
logger = logging.getLogger('my_json')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

@app.route('/')
@tracer.wrap()
def hello_world():
    """
    Example endpoint returning a simple greeting
    ---
    responses:
      200:
        description: A simple greeting message
        examples:
          text: Hello, World!
    """
    logger.info("Hello World endpoint was called")
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
@tracer.wrap()
def predict():
    """
    Predict if a user will churn based on their features
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            messages_sent_count:
              type: integer
              description: Number of messages sent
            messages_received_count:
              type: integer
              description: Number of messages received
            total_minutes_played:
              type: integer
              description: Total minutes played
            average_minutes_per_session:
              type: integer
              description: Average minutes per session
          example:
            messages_sent_count: 5
            messages_received_count: 6
            total_minutes_played: 8
            average_minutes_per_session: 9
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: string
              description: Prediction result
              example: "Churn"
      400:
        description: Invalid input
        schema:
          type: object
          properties:
            error:
              type: string
              description: Error message
              example: "Error message"
    """
    try:
        data = request.get_json(force=True)
        features = pd.DataFrame([{
            'messages_sent_count': data['messages_sent_count'],
            'messages_received_count': data['messages_received_count'],
            'total_minutes_played': data['total_minutes_played'],
            'average_minutes_per_session': data['average_minutes_per_session']
        }])
        prediction = model.predict(features)
        prediction_label = "Churn" if int(prediction[0]) == 1 else "No Churn"
        logger.info(f"Prediction made: {prediction_label}")
        return jsonify({'prediction': prediction_label})
    except KeyError as e:
        logger.error(f"Missing parameter: {str(e)}")
        return jsonify({'error': f"Missing parameter: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009)
