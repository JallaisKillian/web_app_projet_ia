from flask import Flask, request, render_template
from mlp_predict import get_mlp_predict
from svm_predict import get_svm_predict

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

PREDICTIONS = {
    True: "Positif",
    False: "Négatif"
}

@app.route("/predict/", methods=["POST"])
def mlp_predict():
    avis = request.form["avis"]
    model = request.form["model"]
    
    if model == "mlp":
        prediction = get_mlp_predict(avis)
    else:
        prediction = get_svm_predict(avis)
    
    return {"prediction": PREDICTIONS[prediction]}

@app.route("/")
def hello_world():
    return render_template("index.html")
