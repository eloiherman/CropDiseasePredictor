from flask import Flask, render_template, request
from predict import make_prediction

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/try", methods=["GET", "POST"])
def try_model():
    if request.method == "POST":
        prediction, confidence = make_prediction(request.form)
        return render_template("try.html", prediction=prediction, confidence=confidence)

    return render_template("try.html", prediction=None, confidence=None)