import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
load_model = pickle.load(open('NPAfinal_model.sav', 'rb'))


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/predicted", methods=['POST'])
def predicted():
    # FOR RENDING RESULT
    input_news = request.form["news"]
    output = load_model.predict([input_news])

    return render_template('index.html', prediction_text='The News is: {}'.format(output))

def handler(request, response):
    return app(request.environ, start_response=response)

if __name__ == "__main__":
    app.run(debug=True)
