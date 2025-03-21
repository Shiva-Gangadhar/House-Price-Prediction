import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Housing.csv")  # Ensure your dataset file exists

# Select features and target variable
features = ["area", "bedrooms", "bathrooms", "stories", "mainroad", 
            "guestroom", "basement", "airconditioning", "parking", "furnishingstatus"]
target = "price"

X = df[features]
y = df[target]

# Convert categorical features to numerical
X["mainroad"] = X["mainroad"].map({"yes": 1, "no": 0})
X["guestroom"] = X["guestroom"].map({"yes": 1, "no": 0})
X["basement"] = X["basement"].map({"yes": 1, "no": 0})
X["airconditioning"] = X["airconditioning"].map({"yes": 1, "no": 0})
X["furnishingstatus"] = X["furnishingstatus"].map({"furnished": 2, "semi-furnished": 1, "unfurnished": 0})

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully!")
