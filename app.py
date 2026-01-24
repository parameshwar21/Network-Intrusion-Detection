# app.py
import os
from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Load trained SVM model & encoder
# -------------------------------
MODEL_FOLDER = "model"
pipeline = joblib.load(os.path.join(MODEL_FOLDER, "intrusion_model_svm.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_FOLDER, "label_encoder.pkl"))

# -------------------------------
# Feature columns (same as training)
# -------------------------------
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

categorical_cols = ["protocol_type", "service", "flag"]

# -------------------------------
# Home page
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------------------
# Prediction route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Read CSV
    df = pd.read_csv(filepath)

    # Validate columns
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}"

    df = df[columns]

    # Clean categorical columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # -------------------------------
    # Predict (ENCODED → DECODED)
    # -------------------------------
    encoded_preds = pipeline.predict(df)

    # Convert numbers → attack names
    attack_names = label_encoder.inverse_transform(encoded_preds)

    # Add prediction column
    df["Prediction"] = attack_names

    # Save result CSV
    out_file = f"predictions_{file.filename}"
    out_path = os.path.join(UPLOAD_FOLDER, out_file)
    df.to_csv(out_path, index=False)

    return render_template(
        "index.html",
        table_data=df.to_dict(orient="records"),
        columns=df.columns,
        download_file=out_file
    )

# -------------------------------
# Download
# -------------------------------
@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, as_attachment=True)

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
