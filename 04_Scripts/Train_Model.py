# Train Model
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder

# Suppress undefined metric warnings in classification report
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# -------------------- Load Data -------------------- #
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/03_Predictive Maintenance in Manufacturing/02_Data/Preprocessed_data.csv")

# ------------------ Prepare Features ---------------- #
target_col = "Failure Type"
X = df.drop(target_col, axis=1)
y_raw = df[target_col]

print("Original unique classes in y:", y_raw.unique())

# Encode target labels to 0-based integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

num_classes = len(label_encoder.classes_)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -------------- Save Train and Test CSVs -------------- #

# Convert y_train and y_test back to original labels for interpretability
y_train_orig = label_encoder.inverse_transform(y_train)
y_test_orig = label_encoder.inverse_transform(y_test)

# Create DataFrames for train and test sets including target column
train_df = X_train.copy()
train_df[target_col] = y_train_orig

test_df = X_test.copy()
test_df[target_col] = y_test_orig

# Save to CSV
train_df.to_csv("R:/Projects/1_Data_Science & ML_Projects/03_Predictive Maintenance in Manufacturing/06_Outputs/train_data.csv", index=False)
test_df.to_csv("R:/Projects/1_Data_Science & ML_Projects/03_Predictive Maintenance in Manufacturing/06_Outputs/test_data.csv", index=False)

# ------------------ Initialize Models ---------------- #
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    ),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# ------------------ Train & Evaluate ---------------- #
model_scores = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n{name} Accuracy: {acc:.4f}, Macro F1-score: {f1:.4f}")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0,
        target_names=[str(cls) for cls in label_encoder.classes_]
    ))

    model_scores.append((name, model, acc, f1))

# ------------------ Save Top 2 Models ---------------- #
# Sort based on F1-score instead of accuracy (recommended for imbalanced data)
model_scores.sort(key=lambda x: x[3], reverse=True)
top_models = model_scores[:2]

save_dir = "R:/Projects/1_Data_Science & ML_Projects/03_Predictive Maintenance in Manufacturing/05_Models/"
for name, model, acc, f1 in top_models:
    joblib.dump(model, f"{save_dir}{name}_model.pkl")
    print(f"Saved: {name} (Accuracy: {acc:.4f}, F1: {f1:.4f})")

# Save label encoder for decoding predictions later
joblib.dump(label_encoder, f"{save_dir}label_encoder.pkl")

# ------------------ Save Accuracy Summary ---------------- #

# Save model performance summary
with open(f"{save_dir}model_summary.txt", "w") as f:
    for name, _, acc, f1 in model_scores:
        f.write(f"{name} -> Accuracy: {acc:.4f}, Macro F1-score: {f1:.4f}\n")

print("\nTop 2 models (based on Macro F1-score) and summary saved.")