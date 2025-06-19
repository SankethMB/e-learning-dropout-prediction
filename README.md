This project predicts whether a student is likely to drop out or continue based on various academic, demographic, and behavioral features using a machine learning model. The app is built with Flask for the backend and provides an interactive web interface.

📂 Project Structure
e-learning-dropout-prediction/
│
├── model/
│   ├── dropout_model.pkl
│   ├── scaler.pkl
│   ├── le_gender.pkl ... (encoders)
│   └── accuracy.txt
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── train_model_with_encoders.py
├── xAPI-Edu-Data.csv
└── README.md

