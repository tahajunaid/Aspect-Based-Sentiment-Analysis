import os, sys, argparse, logging, re, string
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Concatenate,Dense

from attention import Attention

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

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
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data['text'])
    #saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    tokenized_text = tokenizer.texts_to_sequences(data['text'])
    tokenized_aspect = tokenizer.texts_to_sequences(data['aspect'])
    embedded_text=pad_sequences(tokenized_text,padding='pre',maxlen=120)
    embedded_aspect=pad_sequences(tokenized_aspect,padding='pre',maxlen=8)
    X1=np.array(embedded_text)
    X2=np.array(embedded_aspect)
    y=np.array(data['label'])

    print('\n\nApplying GloVe Embeddings...')
    embeddings_index = {}
    f = open(os.path.join('glove.6B.100d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((5000, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    print('\n\nDone.')
    input_1 = Input(shape=(120,))
    input_2 = Input(shape=(8,))
    embedding_layer_1 = Embedding(5000, 100, weights=[embedding_matrix], trainable=False)(input_1)
    LSTM_Layer_1 = LSTM(128,return_sequences=True)(embedding_layer_1)
    attention_1 = Attention(128)(LSTM_Layer_1)
    embedding_layer_2 = Embedding(5000, 100,weights=[embedding_matrix], trainable=False)(input_2)
    LSTM_Layer_2 = LSTM(32)(embedding_layer_2)
    concat_layer = Concatenate()([attention_1,LSTM_Layer_2])
    dense_layer_1 = Dense(15, activation='relu')(concat_layer)
    output = Dense(3, activation='softmax')(dense_layer_1)
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    X1_train, X1_test,X2_train, X2_test, y_train, y_test = train_test_split(X1,X2, y, test_size=0.1, random_state=42)
    y_train = to_categorical(y_train, 3)
    y_test = to_categorical(y_test, 3)

    print('\n\n--------------------------------------------Training Started.--------------------------------------------')
    history = model.fit(x=[X1_train, X2_train], y=y_train, batch_size=72, epochs=5, verbose=1, validation_split=0.1)
    loss,accuracy=model.evaluate([X1_test,X2_test],y_test)
    print('\n\n--------------------------------------------Training Completed.--------------------------------------------')

    print('\n\nAccuracy: {}'.format(round(accuracy,2)))
    print('Loss: {}\n\n'.format(round(loss,2)))

    y_pred_probs=model.predict([X1_test,X2_test])
    y_pred=np.argmax(y_pred_probs,axis=1)
    y_actual=np.argmax(y_test,axis=1)
    labels=[0,1,2]
    print('\n\n--------------------------------------------Classification Report--------------------------------------------')
    print(classification_report(y_actual.tolist(),y_pred.tolist(),labels=labels))

    model.save('test_model')

if __name__ == "__main__":
	main(sys.argv[1:])
