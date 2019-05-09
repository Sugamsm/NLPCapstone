from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#import nltk
import json
from keras import backend as K

#K.clear_session()

def saveDict(name, location, json):
    f = open(name, 'w')
    f.write(json)
    f.close()

def getPoemPrediction(inp, model):
    js = json.loads(open('S:/SUNY/Semester 4/Project/Code/poem/poem_data_dict.json', 'r').read())
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

def getArticlePrediction(inp, model):
    inp = inp.lower()    
    js = json.loads(open('S:/SUNY/Semester 4/Project/Code/articles/article_data_dict.json', 'r').read())
    w_to_n = js['word_ind_dict']
    n_to_w = js['ind_word_dict']
    longest_sequence = js['long_seq']
    for i in range(30):
        words = inp.split(' ')
        sequences = [w_to_n[word] for word in words]
        sequences = pad_sequences([sequences], maxlen=longest_sequence, padding='pre')
        sequences = sequences.reshape(1, 1, longest_sequence)
        predicted = model.predict_classes(sequences)
        for word, index in w_to_n.items():
            if index == predicted:
                inp += " " + word
                break
    return inp
