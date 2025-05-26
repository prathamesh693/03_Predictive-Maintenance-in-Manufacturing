# Predict
import pandas as pd
import joblib
import os

# Define model paths (adjust if needed)
model_dir = "R:/Projects/1_Data_Science & ML_Projects/03_Predictive Maintenance in Manufacturing/05_Models/"
models_to_load = ["RandomForest_model.pkl", "XGBoost_model.pkl"]

# Load input data for prediction
input_data = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/03_Predictive Maintenance in Manufacturing/02_Data/Preprocessed_data.csv")

# Drop target column if exists
if "Failure Type" in input_data.columns:
    input_data = input_data.drop("Failure Type", axis=1)

# Map encoded predictions back to failure type labels (if known mapping)
label_mapping = {
    0: "No Failure",
    1: "Heat Dissipation Failure",
    2: "Power Failure",
    3: "Tool Wear Failure",
    4: "Overstrain Failure",
    5: "Random Failures"
}

# Create results DataFrame
results = pd.DataFrame(input_data.copy())

# Predict using top 2 models
for model_file in models_to_load:
    model_path = os.path.join(model_dir, model_file)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        preds = model.predict(input_data)
        decoded_preds = [label_mapping.get(p, "Unknown") for p in preds]
        model_name = model_file.replace("_model.pkl", "")
        results[f"Prediction ({model_name})"] = decoded_preds
        print(f"{model_name} predictions complete.")
    else:
        print(f"Model file {model_file} not found!")

# Save predictions
results.to_csv("R:/Projects/1_Data_Science & ML_Projects/03_Predictive Maintenance in Manufacturing/06_Outputs/prediction_result.csv", index=False)
print("\nPredictions saved to Output folder!")
