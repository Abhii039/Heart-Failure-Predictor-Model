from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load your model and scaler
model = joblib.load('heart_failure_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the saved scaler

app = Flask(__name__)

# Enable CORS
CORS(app)

@app.route('/',methods=['GET'])
def default():
    return 'Server Started.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        age = float(data['age'])
        anaemia = int(data['anaemia'])
        creatinine_phosphokinase = float(data['creatininePhosphokinase'])
        diabetes = int(data['diabetes'])
        ejection_fraction = float(data['ejectionFraction'])
        high_blood_pressure = int(data['highBloodPressure'])
        platelets = float(data['platelets'])
        serum_creatinine = float(data['serumCreatinine'])
        serum_sodium = float(data['serumSodium'])
        sex = int(data['sex'])
        smoking = int(data['smoking'])
        time = float(data['time'])
    except KeyError as e:
        return jsonify({'error': f'Missing data for {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid data type: {str(e)}'}), 400

    # Prepare features for prediction
    features = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                          high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex,
                          smoking, time]])

    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)

    # Return prediction as JSON
    result = 'Death Event' if int(prediction[0]) == 1 else 'Survived'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
