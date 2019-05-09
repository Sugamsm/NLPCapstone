from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
import json
from keras import backend as K

K.clear_session()

def saveDict(name, location, json):
    f = open(name, 'w')
    f.write(json)
    f.close()

def getPoemPrediction(inp, model):
    js = json.loads(open('./poem/poem_data_dict.json', 'r').read())
    word_dict = dict(js['word_ind_dict'])
    tokenizer = Tokenizer()
    tokenizer.word_index = word_dict
    for i in range(30):
        words = tokenizer.texts_to_sequences([inp])[0]
        #words = [word_dict[word] for word in words]
        words = pad_sequences([words], maxlen=js['long_seq'], padding='pre')
        predicted = model.predict_classes(words)
        print(predicted)
        for word, index in word_dict.items():
            if index == predicted:
                print(word)
                inp += " " + word
                break
    return inp
