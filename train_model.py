import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
df = pd.read_csv("xAPI-Edu-Data.csv")

# Create target column: Dropout if Class == 'L'
df['Dropout'] = df['Class'].apply(lambda x: 1 if x == 'L' else 0)

# Define categorical features
cat_features = [
    'gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
    'SectionID', 'Topic', 'Semester', 'Relation',
    'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays'
]

# Create directory to save encoders and model
os.makedirs("model", exist_ok=True)

# Encode categorical features and save encoders
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    pickle.dump(le, open(f"model/le_{col}.pkl", "wb"))

# Define features and target
features = cat_features + ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
X = df[features]
y = df['Dropout']

# Scale numerical features and save scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate and save accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
with open("model/accuracy.txt", "w") as f:
    f.write(f"{accuracy * 100:.2f}")

# Save model
pickle.dump(model, open("model/dropout_model.pkl", "wb"))

print(f"✅ Model trained successfully with accuracy: {accuracy * 100:.2f}%")
print("✅ Encoders, scaler, model saved in 'model/' folder.")
