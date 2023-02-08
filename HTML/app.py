from flask import Flask
from flask import render_template
from flask import request
from flask import abort, redirect, url_for
from PIL import Image
import ml

app = Flask(__name__)


@app.route('/')
@app.route('/<name>')
def hello(name=None):
    return render_template('index.html', name=name)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('./uploads/' + f.filename)

        upload = Image.open('./uploads/' + f.filename)
        upload = upload.convert("RGBA")

        filename = b"../linear_model_save.txt"
        W_flag = ml.LoadModeleLineaire(filename)
        predictml = ml.PredictModeleLineaire(ml.averageRGB100(upload), W_flag, True)

        gamma = 0.0000001
        filename = b"../rbf_save.txt"
        W, uks = ml.LoadRBF(filename)
        predictrbf = ml.PredictRBF(ml.averageRGB100(upload), W, True, gamma, uks)

        filename = b"../test.txt"
        pmc_flag = ml.CreatePMCFromFile(filename)
        predictpmc = ml.PredictPMC(pmc_flag, ml.averageRGB100(upload), True)
        ml.FreePMC(pmc_flag)

    return render_template('index.html', predictml=predictml, predictrbf=predictrbf, predictpmc=predictpmc)
