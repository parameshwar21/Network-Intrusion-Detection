# predict.py
import os
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Paths to saved model & encoder
# -------------------------------
model_folder = os.path.join(os.getcwd(), "model")
model_path = os.path.join(model_folder, "intrusion_model.pkl")
encoder_path = os.path.join(model_folder, "label_encoder.pkl")

# Load model & encoder
pipeline = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# -------------------------------
# Column names & features
# -------------------------------
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

categorical_cols = ["protocol_type", "service", "flag"]
numerical_cols = [c for c in columns if c not in categorical_cols + ["label", "difficulty"]]

# -------------------------------
# Function to predict a single row
# -------------------------------
def predict_single(row_dict):
    """
    row_dict: dictionary of feature_name: value
    returns: predicted label
    """
    df = pd.DataFrame([row_dict])
    
    # Ensure columns exist
    for col in categorical_cols + numerical_cols:
        if col not in df.columns:
            df[col] = 0  # fill missing numeric features with 0
        if col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()
    
    # Predict using pipeline
    pred_enc = pipeline.predict(df[categorical_cols + numerical_cols])
    pred_label = label_encoder.inverse_transform(pred_enc)
    return pred_label[0]

# -------------------------------
# Function to predict from CSV file
# -------------------------------
def predict_from_csv(file_path):
    df = pd.read_csv(file_path)
    for col in categorical_cols + numerical_cols:
        if col not in df.columns:
            df[col] = 0
        if col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()
    
    pred_enc = pipeline.predict(df[categorical_cols + numerical_cols])
    df["predicted_label"] = label_encoder.inverse_transform(pred_enc)
    return df

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example single row
    example = {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 181, "dst_bytes": 5450, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 9, "srv_count": 9, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 9,
        "dst_host_srv_count": 9, "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    }
    
    label = predict_single(example)
    print(f"Predicted label for single row: {label}")
    
    # Example CSV prediction
    # result_df = predict_from_csv("dataset/sample_test.csv")
    # print(result_df.head())
