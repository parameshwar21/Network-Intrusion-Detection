Network Intrusion Detection Using Machine Learning

A Flask web application that allows users to detect network intrusions from CSV files using a trained machine learning model. The application predicts whether each network connection is normal or an attack type (DoS, Probe, R2L, U2R) and displays results in the browser.

Features

Upload CSV files with network data.

Predict intrusions using a Random Forest classifier.

Display predictions in a browser table.

Highlight suspicious/attack entries.

Save trained model and label encoders for reuse.

Dataset

Based on NSL-KDD dataset or your own network traffic dataset.

Includes 41+ network traffic features such as duration, protocol_type, service, flag, src_bytes, dst_bytes, and more.

Features are encoded using LabelEncoder before feeding into the model.

Technologies

Python 3.10+

Flask – for web application

pandas – for data handling

scikit-learn – for model training and prediction

pickle – for saving/loading trained model

Installation

1.Clone this repository:

git clone <your-repo-url>
cd network-intrusion-detection


2.Create a virtual environment (recommended):

python -m venv nids_env
nids_env\Scripts\activate   # Windows
# source nids_env/bin/activate  # Linux/Mac


3.Install dependencies:

pip install numpy pandas scikit-learn flask

Usage

Train the model (if you don’t have a pre-trained .pkl):

python retrain_save_model.py


Run the Flask app:

python app.py


Open your browser and go to:

http://127.0.0.1:5000


Upload CSV files and click Predict to see results.

File Structure
network-intrusion-detection/
│
├─ app.py                  # Main Flask application
├─ predict.py              # Prediction functions using trained model
├─ retrain_save_model.py   # Script to train & save the model locally
├─ model/                  # Folder containing model & label encoders
│   ├─ intrusion_model.pkl
│   └─ label_encoders.pkl
├─ templates/
│   └─ index.html          # Web page template for file upload & results
├─ static/
│   └─ styles.css          # Optional: CSS for table styling
├─ Train_data.csv          # Sample training dataset
├─ Test_data.csv           # Sample testing dataset
└─ README.md

Model Training

The project uses a Random Forest Classifier from scikit-learn.

Categorical features (protocol_type, service, flag, etc.) are encoded with LabelEncoder.

Model and encoders are saved using pickle for later use in Flask.

Retraining ensures compatibility with your current numpy/scikit-learn versions.

CSV Format

CSV must include all 41+ features used during training.

Column names must match the training dataset exactly.

Example columns:

duration,protocol_type,service,flag,src_bytes,dst_bytes,count,srv_count,...,label
