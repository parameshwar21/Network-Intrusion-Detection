import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# 1. Load Dataset
# ==========================
train_path = "Train_data.csv"  # update path if needed
test_path  = "Test_data.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ==========================
# 2. Identify Target Column
# ==========================
TARGET_COL = train_df.columns[-1]  # usually the last column
print("Target column:", TARGET_COL)

# ==========================
# 3. Encode Categorical Columns
# ==========================
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

print("Categorical features encoded.")

# ==========================
# 4. Encode Target Column
# ==========================
le_target = LabelEncoder()
train_df[TARGET_COL] = le_target.fit_transform(train_df[TARGET_COL])
label_encoders['label'] = le_target

print("Target column encoded.")

# ==========================
# 5. Split Features & Target
# ==========================
X_train = train_df.drop(TARGET_COL, axis=1)
y_train = train_df[TARGET_COL]

X_test = test_df.drop(TARGET_COL, axis=1, errors='ignore')  # some test CSV may not have target
y_test = test_df[TARGET_COL] if TARGET_COL in test_df.columns else None

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# ==========================
# 6. Train Random Forest Model
# ==========================
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model trained successfully.")

# ==========================
# 7. Evaluate Model (if labels exist in test)
# ==========================
if y_test is not None:
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==========================
# 8. Save Model & Encoders
# ==========================
os.makedirs("model", exist_ok=True)

with open("model/intrusion_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model and label encoders saved in 'model/' folder successfully.")
