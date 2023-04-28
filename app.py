import pandas as pd
import numpy as np

import os
import sys

import random
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from IPython.display import Audio

from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM, Bidirectional
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

!pip install librosa

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
    result = extarct_features(testMale)
    features=np.expand_dims(result,axis=0)
    features=np.expand_dims(result,axis=2)
    pred = loaded_model.predict(result)
    return pred

from flask import Flask, request 
from werkzeug.utils import secure_filename
#Create a new Flask app
app = Flask(__name__) 

@app.route('/hello')
def hello_world():
    return 'Hello World!'

#Create a new route for the upload endpoint with the POST method
@app.route('/upload', methods=['POST'])
def upload_audio():
    audio = request.files['audio'] 
    filename = secure_filename(audio.filename)
    audio.save(filename)
    print('here')
    audioClass = audio_class(audio)
    print(audioClass)
    return 'Audio received' 

if __name__ == '__main__':
    app.run() 
