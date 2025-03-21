import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# Create Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from request
        data = request.get_json()

        # Convert categorical values to numerical
        mainroad = 1 if data["mainroad"] == "yes" else 0
        guestroom = 1 if data["guestroom"] == "yes" else 0
        basement = 1 if data["basement"] == "yes" else 0
        airconditioning = 1 if data["airconditioning"] == "yes" else 0

        furnishingstatus = data["furnishingstatus"]
        if furnishingstatus == "furnished":
            furnishingstatus = 2
        elif furnishingstatus == "semi-furnished":
            furnishingstatus = 1
        else:
            furnishingstatus = 0

        # Prepare input for the model
        features = np.array([
            float(data["area"]),
            int(data["bedrooms"]),
            int(data["bathrooms"]),
            int(data["stories"]),
            mainroad,
            guestroom,
            basement,
            airconditioning,
            int(data["parking"]),
            furnishingstatus
        ]).reshape(1, -1)

        # Scale the input data
        features_scaled = scaler.transform(features)

        # Make prediction
        predicted_price = model.predict(features_scaled)[0]

        return jsonify({"predicted_price": float(predicted_price)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
