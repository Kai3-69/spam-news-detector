from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    vect = vectorizer.transform([news])
    prediction = model.predict(vect)

    if prediction[0] == 1:
        result = "Real News"
    else:
        result = "Spam / Fake News"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)