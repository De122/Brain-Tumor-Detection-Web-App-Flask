from flask import Flask,request,render_template,redirect,url_for,flash
import h5py
import tensorflow  as tf
from keras.preprocessing import image
from keras.models import load_model
import keras.utils as image
from PIL import Image
import numpy as np 
import os
from werkzeug.utils import secure_filename
import urllib.request

app=Flask(__name__)


model=load_model(r"D:\WebApp_BrainTumorDetection\test\Tumor_classifier_model.h5")


def load_image(img_path):
    i=image.load_img(img_path,target_size=(224,224))
    i=image.img_to_array(i)/255
    x=np.expand_dims(i,axis=0)
    return x

def predict_label(img_path):
    n_image=load_image(img_path)
    pred=model.predict(n_image)
    labels=np.array(pred)
    labels[labels>=0.6]=1
    labels[labels<0.6]=0
    f=np.array(labels)
    if f[0][0]==1:
        return 'Yes,Brain tumor detected !'
    else:
        return 'No Brain tumor detected!'

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/submit',methods=['GET','POST'])
def result():
    if request.method=='POST':
        
        img=request.files['MRI']
        imgname=img.filename
        img_path=os.path.join(r'D:\WebApp_BrainTumorDetection\archive\brain_tumor_dataset',imgname)
        img.save(img_path)
        print(imgname)
        p=predict_label(img_path)
        print(p)
    return render_template('result.html',prediction=p,img_path=img_path)


    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)








