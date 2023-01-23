import json
from flask import Flask, jsonify, render_template, request , send_from_directory
import numpy as np
from keras.models import load_model 
import cv2
from flask_cors import cross_origin , CORS
import pandas as pd
import tensorflow as tf
from flask_swagger_ui import get_swaggerui_blueprint



app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

### swagger specific ###
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Classification de déchets à la ville de Paris"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
### end swagger specific ###


loaded_model = tf.keras.models.load_model('best_weights.hdf5')


COUNT = 0
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def main():
    return render_template('index.html')



DATASET_COMPOSTABLE = "DATASET_TRASH/dataset_compostable.csv"
DATASET_COMPOSTABLE_COULEURS = "DATASET_TRASH/dataset_compostable2.csv"
DATASET_RECYCLABLE = "DATASET_TRASH/dataset_recyclable.csv"

# Tri
@app.route('/tri', methods=['POST','GET'])
@cross_origin()
def home():
    global COUNT
    img = request.files['image']
    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224,224,3)
    prediction = loaded_model.predict(img_arr)

    x = prediction[0,0]
    y = 1-x
    preds = np.array(x)
    preds_y = np.array(y)
    preds = json.dumps(preds.tolist())
    preds_y = json.dumps(preds_y.tolist())
    COUNT += 1
    print(preds)
    print(preds_y)

    code = request.form.get("code")
    if not code:
        return jsonify({"error": "Code postal manquant"})

    if isinstance(code, str):
        code = int(code)
    if float(preds) > 0.5:
        poubelle = 'yellow'
        message = 'recyclable'
        preds = preds
        dataset = pd.read_csv(DATASET_RECYCLABLE)
    else:
        dataset_couleur = pd.read_csv(DATASET_COMPOSTABLE_COULEURS)
        dataset_couleur["Code postal"] = dataset_couleur["Code postal"].astype(str).str.replace(".", "").astype(int)
        if code in set(dataset_couleur["Code postal"]):
            poubelle = list(dataset_couleur[dataset_couleur["Code postal"] == code]["couleur"])[0]
        else:
            poubelle = "not found"
        message = 'organic'
        preds = preds_y
        dataset = pd.read_csv(DATASET_COMPOSTABLE)
    
    dataset["Code postal"] = dataset["Code postal"].astype(str).str.replace(".", "").astype(int)
    if code in set(dataset["Code postal"]):
        adresses = list(dataset[dataset["Code postal"] == code]["Adresse"])
        return jsonify({"colorTrash": poubelle, "probability": preds, "typeTrash": message, "adresses": adresses})
    else:
        return jsonify({"colorTrash": poubelle, "probability": preds, "typeTrash": message, "adresses": []})



@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')