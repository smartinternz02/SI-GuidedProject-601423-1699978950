import os
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained KNN model
model_path = os.environ.get('MODEL_PATH', 'knn_model5.pkl')
model = pickle.load(open(model_path, 'rb'))


@app.route('/')
def index():
     return ('index.html')


# Define an endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request in JSON format
        data = request.get_json()

        # Ensure that the input data matches the expected features
        expected_features = ['Airline Name', 'Overall Rating', 'Verified', 'Type Of Traveller',
                      'Seat Type', 'origin', 'destination', 'monthFlown', 'yearFlown',
                      'seatComfort', 'cabinStaffService', 'foodBeverages', 'groundService']


        for feature in expected_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Prepare the input data for prediction
        input_data = []
        for feature in expected_features:
            input_data.append(data[feature])

        # Make the prediction using the loaded model
        prediction = model.predict([input_data])[0]
        
        recommendation = 'Recommended' if prediction == 1 else 'Not Recommended'

        # Return the prediction as JSON response
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
