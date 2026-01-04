import pandas as pd
import pickle

# Load trained model and label encoders
with open("model/intrusion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

def predict_intrusion_file(file_path):
    df = pd.read_csv(file_path)

    # Drop target if present
    if 'label' in df.columns:
        df = df.drop('label', axis=1)
    if 'class' in df.columns:
        df = df.drop('class', axis=1)

    # Fill missing features and drop extra features
    model_features = model.feature_names_in_
    missing_cols = set(model_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    extra_cols = set(df.columns) - set(model_features)
    df = df.drop(columns=list(extra_cols))
    df = df[model_features]

    # Encode categorical columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    predictions = model.predict(df)
    le_label = label_encoders['label']
    df['prediction'] = le_label.inverse_transform(predictions)

    return df
