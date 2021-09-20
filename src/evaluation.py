import tensorflow as tf
import pandas as pd
import numpy as np
import os, re, sys, argparse, logging
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer,one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.get_logger().setLevel('ERROR')


def process_args(args):
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', dest="data",type=str)
	parameters = parser.parse_args(args)
	return parameters

def main(args):
    parameters = process_args(args)
    data=pd.read_csv(parameters.data)
    def preprocess_text(sen):
        # Remove non alphanumeric characters
        sentence = re.sub('[^0-9a-zA-Z]', ' ', sen)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    data['text'] = data['text'].apply(preprocess_text)
    data['aspect'] = data['aspect'].apply(preprocess_text)
    # loading tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    tokenized_text = tokenizer.texts_to_sequences(data['text'])
    tokenized_aspect = tokenizer.texts_to_sequences(data['aspect'])
    embedded_text=pad_sequences(tokenized_text,padding='pre',maxlen=120)
    embedded_aspect=pad_sequences(tokenized_aspect,padding='pre',maxlen=8)
    X1=np.array(embedded_text)
    X2=np.array(embedded_aspect)
    parent_dir=os.path.abspath(os.path.join('..'))
    model = tf.keras.models.load_model(parent_dir+'\models\ABSA_Model_5')
    y_pred_probs=model.predict([X1,X2])
    y_pred=np.argmax(y_pred_probs,axis=1)
    res = {'Text':data['text'].tolist(),
            'Aspect':data['aspect'].tolist(),
            'Polarity':y_pred}
    result = pd.DataFrame(res)

    result.to_csv(parent_dir+'\\data\\results\\test.csv')
    print(result)


if __name__ == "__main__":
	main(sys.argv[1:])