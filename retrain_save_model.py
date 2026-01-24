# retrain_save_model_svm.py
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# -------------------------------
# Paths
# -------------------------------
dataset_folder = os.path.join(os.getcwd(), "dataset")  # dataset folder
train_path = os.path.join(dataset_folder, "KDDTrain.txt")
test_path  = os.path.join(dataset_folder, "KDDTest.txt")
model_folder = os.path.join(os.getcwd(), "model")
os.makedirs(model_folder, exist_ok=True)

# -------------------------------
# Column names for NSL-KDD
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

# -------------------------------
# Load datasets
# -------------------------------
def load_dataset(path):
    try:
        df = pd.read_csv(path, names=columns, sep="\t", engine="python")
        if df.shape[1] == 1:  # fallback to comma if tab failed
            df = pd.read_csv(path, names=columns, sep=",", engine="python")
    except Exception as e:
        raise ValueError(f"Error loading {path}: {e}")
    return df

train_df = load_dataset(train_path)
test_df  = load_dataset(test_path)

# -------------------------------
# Clean string columns
# -------------------------------
for col in train_df.select_dtypes(include="object").columns:
    train_df[col] = train_df[col].astype(str).str.strip()
for col in test_df.select_dtypes(include="object").columns:
    test_df[col] = test_df[col].astype(str).str.strip()

# -------------------------------
# Features & labels
# -------------------------------
categorical_cols = ["protocol_type", "service", "flag"]
numerical_cols   = [c for c in columns if c not in categorical_cols + ["label", "difficulty"]]

X_train = train_df[categorical_cols + numerical_cols]
y_train = train_df["label"]

X_test  = test_df[categorical_cols + numerical_cols]
y_test  = test_df["label"]

# -------------------------------
# Encode categorical features
# -------------------------------
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# -------------------------------
# Encode labels safely
# -------------------------------
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)

# Map unseen test labels to -1
y_test_enc = np.array([
    label_encoder.transform([lbl])[0] if lbl in label_encoder.classes_ else -1
    for lbl in y_test
])

# -------------------------------
# Build pipeline with SVM
# -------------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
])

# -------------------------------
# Train model
# -------------------------------
print("Training SVM model...")
pipeline.fit(X_train, y_train_enc)
print("Training completed!")

# -------------------------------
# Save model & label encoder
# -------------------------------
joblib.dump(pipeline, os.path.join(model_folder, "intrusion_model_svm.pkl"))
joblib.dump(label_encoder, os.path.join(model_folder, "label_encoder.pkl"))
print(f"SVM model and label encoder saved in '{model_folder}'")
