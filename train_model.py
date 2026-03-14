import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

data = data.sample(frac=1).reset_index(drop=True)

# Combine title and text
data["content"] = data["title"] + " " + data["text"]

def clean_text(text):

    text = text.lower()
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\s+',' ',text)

    return text

data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=50000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=2000)

model.fit(X_train_vec, y_train)

# Predictions
predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

print(classification_report(y_test, predictions))

# Save model
joblib.dump(model,"model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")

print("Model saved successfully")