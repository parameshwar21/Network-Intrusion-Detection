from flask import Flask, render_template, request
from predict import predict_intrusion_file
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Predict using your model
    result_df = predict_intrusion_file(file_path)

    # Convert to HTML table
    table_html = result_df.to_html(classes='table table-striped', index=False)
    return render_template('index.html', table_html=table_html)

if __name__ == '__main__':
    app.run(debug=True)
