from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder='statics')

# Load trained model
model = tf.keras.models.load_model('breast_cancer_model.keras')

# Binary class labels
class_labels = ['Malignant', 'Benign']


@app.route('/')
def home():
    return "Welcome to the Breast Cancer Prediction API!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form

            # Expecting 30 input features
            features = []
            for i in range(30):
                features.append(float(data[f'feature_{i}']))

            # Convert to NumPy array
            input_data = np.array(features)[np.newaxis, :]

            # Make prediction
            prediction = model.predict(input_data)

            # Convert sigmoid output to class
            predicted_class = class_labels[1] if prediction[0][0] > 0.5 else class_labels[0]

            return jsonify({
                "predicted_class": predicted_class,
                "confidence": float(prediction[0][0])
            })

        except Exception as e:
            return jsonify({"error": str(e)})

    elif request.method == 'GET':
        return render_template('predict.html')

    else:
        return "Unsupported HTTP method"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
