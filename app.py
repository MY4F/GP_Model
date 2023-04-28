import pandas as pd
import numpy as np

import librosa


from IPython.display import Audio

from tensorflow import keras
from keras.models import model_from_json


json_file = open('Gender_Classification_conv1d_Model_99.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("Gender_Classification_conv1d_Model_99.h5")
print("Loaded model from disk")
 
# Keras optimiser
loaded_model.compile(optimizer = 'RMSprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
# score = loaded_model.evaluate(x_test, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

def extract_features(data):
    
    result = np.array([])
    
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
     
    return result

def audio_class(audio_path):
    testMale, sr = librosa.load(audio_path) 
    result = extract_features(testMale)
    features=np.expand_dims(result,axis=0)
    print(features.shape)
    features=np.expand_dims(features,axis=2)
    print(features.shape)
    pred = loaded_model.predict(features)
    return pred

from flask import Flask, request 
from werkzeug.utils import secure_filename
#Create a new Flask app
app = Flask(__name__) 
#Create a new route for the upload endpoint with the POST method
@app.route('/upload', methods=['POST'])
def upload_audio():
    audio = request.files['audio'] 
    filename = secure_filename(audio.filename)
    audio.save(filename)
    audioClass = audio_class(filename)
    print(np.argmax(audioClass))
    if np.argmax(audioClass) == 0:
        return "female"
    else:
        return "male"

if __name__ == '__main__':
    app.run() 
