from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model, scaler, and accuracy
model = pickle.load(open('model/dropout_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))


# Categorical fields and encoders
categorical_fields = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
                      'SectionID', 'Topic', 'Semester', 'Relation',
                      'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']

encoders = {}
for field in categorical_fields:
    encoders[field] = pickle.load(open(f'model/le_{field}.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Categorical inputs
        cat_values = []
        for field in categorical_fields:
            value = request.form.get(field)
            if value not in encoders[field].classes_:
                return f"⚠️ Error: '{value}' is not a recognized value for {field}.", 400
            encoded = encoders[field].transform([value])[0]
            cat_values.append(encoded)

        # Numerical inputs
        raisedhands = int(request.form.get('raisedhands'))
        VisITedResources = int(request.form.get('VisITedResources'))
        AnnouncementsView = int(request.form.get('AnnouncementsView'))
        Discussion = int(request.form.get('Discussion'))

        # Combine features
        feature_values = cat_values + [raisedhands, VisITedResources, AnnouncementsView, Discussion]
        all_columns = categorical_fields + ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
        input_df = pd.DataFrame([feature_values], columns=all_columns)

        # Scale and predict
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        result = '🎓 Likely to Dropout' if prediction == 1 else '✅ Not Likely to Dropout'
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"⚠️ Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
