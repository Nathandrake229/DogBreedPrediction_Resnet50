import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from flask import json
import math
from flask import Flask, request, Response
import keras
import tensorflow as tf
from keras.models import load_model

import base64
import numpy as np
from PIL import Image 

app = Flask(__name__)


np.set_printoptions(suppress=True)

li = ['beagle', 'chihuahua', 'doberman', 'french_bulldog', 'golden_retriever', 'malamute', 'pug', 'saint_bernard', 'scottish_deerhound', 'tibetan_mastiff']

#Reading the decoded image sending it to Resnet50 for dog breed prediction
def classifier(path):
    imag=cv2.imread("C:/work/download_decode.jpg" ,cv2.IMREAD_COLOR)
    imag=cv2.resize(imag,(224, 224))
    test = []
    test.append(imag)
    test = np.array(test)
    model = keras.models.load_model("resnet50_dog_model_3.h5")
    ans = model.predict(test)
    ans = ans.tolist()
    prob = max(ans[0])
    breed = li[ans[0].index(prob)]
    return breed, prob


#Setting up POST route which receives JSON file with base64 encoded image string and returns the dog breed in that image along with its probablity
@app.route('/predict', methods=['POST']) 
def upload_base64_file(): 
    
    data = request.get_json()

    if data is None:
        print("No valid request body, json missing!")
        return jsonify({'error': 'No valid request body, json missing!'})
    else:
        #decoding base64 string into RGB Image and saving locally
        img_data = data['image']
        img_data = str(img_data.split(',')[1])
        image_64_decode = base64.decodestring(img_data.encode())
        image_result = open('C:/work/download_decode.jpg', 'wb') 
        image_result.write(image_64_decode)
        image_result.close()

        #Sending the decoded image path to the classifier function for Dog Breed Classification
        breed, prob = classifier('C:/work/download_decode.jpg')
        
        #Removing the image after classification is complete
        os.remove('C:/work/download_decode.jpg')
    return {'breed': breed, 'score': prob}





if __name__ == "__main__":
    app.run(port=1234, debug=True)