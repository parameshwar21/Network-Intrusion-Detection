Network Intrusion Detection System (NIDS)

Project Overview:
The Network Intrusion Detection System (NIDS) is a cybersecurity project designed to monitor network traffic and detect suspicious activities, intrusions, or attacks in real-time. 
Using machine learning techniques, the system can classify network traffic as normal or malicious, helping organizations prevent potential security breaches.



Project Structure
network-intrusion-detection/
│
├── data/                  # Dataset files
├── models/                # Trained ML models
├── results/               # Generated reports and logs
├── train_model.py         # Script to train ML models
├── test_model.py          # Script to test or monitor traffic
├── main.py                # Entry point of the project
├── config.py              # Configuration file
├── requirements.txt       # Python dependencies
└── README.md

Installation
Clone the repository:
git clone https://github.com/parameshwar21/network-intrusion-detection.git

Navigate to the project folder:
cd network-intrusion-detection

Install the required dependencies:
pip install -r requirements.txt


Usage

Prepare your dataset (e.g., NSL-KDD) in CSV format.

Configure dataset path in the config.py file.

Train the machine learning model:

python train_model.py


Test or monitor network traffic:

python test_model.py


View reports and results in the results/ folder.


Run the project:

python main.py
