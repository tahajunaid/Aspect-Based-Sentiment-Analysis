import tensorflow as tf
import pandas as pd
import numpy as np
import os, re, sys, argparse, logging
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer,one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict(text,aspect):
    def preprocess_text(sen):
        # Remove non alphanumeric characters
        sentence = re.sub('[^0-9a-zA-Z]', ' ', sen)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    text = preprocess_text(text)
    aspect = preprocess_text(aspect)
    # loading tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenized_text=np.array(list(np.concatenate(tokenizer.texts_to_sequences(text.split())).flat)).reshape(1,-1)
    tokenized_aspect=np.array(list(np.concatenate(tokenizer.texts_to_sequences(aspect.split())).flat)).reshape(1,-1)

    embedded_text=pad_sequences(tokenized_text,padding='pre',maxlen=120)
    embedded_aspect=pad_sequences(tokenized_aspect,padding='pre',maxlen=8)

    X1=embedded_text
    X2=embedded_aspect

    parent_dir=os.path.abspath(os.path.join('..',os.getcwd()))
    model = tf.keras.models.load_model(parent_dir+'/models/ABSA_Model_5')
    y_pred_probs=model.predict([X1,X2])
    y_pred=np.argmax(y_pred_probs)
    polarity_dict = {'0' : 'Negative','1' : 'Neutral','2' : "Positive"}
    polarity=polarity_dict.get(str(y_pred), -1)
    return polarity
