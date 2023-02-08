from flask import render_template, request
from flask import current_app as app
from model.model_prediction import predict

@app.route("/", methods = ["GET", "POST"])
def home():
    op = -1
    if request.method == "POST":
        all_crops = ["Wheat", "Ground Nuts", "Garden flowers", "Maize", "Paddy", "Potato", "Pulse", "SugerCane", "Coffee"]
        ct = all_crops.index(request.form["cropType"]) + 1
        cd = 0 if request.form["cropdays"] == "" else int(request.form["cropdays"])
        moisture = 0 if request.form["moisture"] == "" else int(request.form["moisture"])
        temperature = 0 if request.form["temperature"] == "" else int(request.form["temperature"])
        humidity = 0 if request.form["humidity"] == "" else int(request.form["humidity"])
        op = predict(ct, cd, moisture, temperature, humidity)

    return render_template("index.html", output = op)