from flask import Flask, request, render_template
from flask_cors import CORS
from flask_restful import Api
import os
import secrets
import pandas as pd
import numpy as np
import sys
from flask import Flask
from knn import KNN
from glcm import GLCM
import base64


# init object flask
app = Flask(__name__)

# init object flask restfull
api = Api(app)

# init cors
CORS(app)

df_fitur_train = pd.read_csv("python/train_glcm_feature_training_rev.csv")
df_label_train = df_fitur_train.pop("labels")
df_fitur_train = np.array(df_fitur_train)
df_label_train = np.array(df_label_train)
model  = KNN(k=17)
model.train(df_fitur_train,df_label_train)
glcm = GLCM()

def base64_to_image(base64_data):
    unique = secrets.token_urlsafe(4)
    img_data = base64_data
    img_data = img_data.split("base64")[-1]
    img_data = bytes(img_data, encoding='utf-8')
    converted = str(os.path.join('./classify/',unique+'-.png'))
    with open(converted, "wb") as fh:
        fh.write(base64.decodebytes(img_data))
    return converted

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/prediksigambar", methods=["POST"])
def processimages():
    global glcm 
    global model

    gambar = request.files["gambar"]
    unique = secrets.token_urlsafe(4)
    imgDir = os.path.join('./classify/',unique+'-'+gambar.filename)
    gambar.save(imgDir)
        
    img = imgDir
    Eng,diss,hom,idm,ent,Asm = glcm.get_feature(img)
    fitur_test = [Eng,diss,hom,idm,ent,Asm]
    fitur_test = np.reshape(fitur_test,(1,-1))
    result = model.predict(fitur_test)[0]
    os.remove(img)
    return result

@app.route("/prediksi", methods=["POST"])
def processfiles():  
    global glcm 
    global model
    
    gambar = request.form["gambar"]

    # img = "python\dataset\datatesting\kecubung\kecubung1.png"
    img = base64_to_image(gambar)
    Eng,diss,hom,idm,ent,Asm = glcm.get_feature(img)
    fitur_test = [Eng,diss,hom,idm,ent,Asm]
    fitur_test = np.reshape(fitur_test,(1,-1))
    result = model.predict(fitur_test)[0]
    os.remove(img)
    return result

if __name__=='__main__':
    app.run()