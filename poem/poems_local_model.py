from keras.preprocessing.text import Tokenizer
import keras.utils as k_utils
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Dropout, Embedding
import numpy as np
import re
#import nltk
#nltk.download()
#reading data

poems = open('S:/SUNY/Semester 4/Project/Capstone/Capstone/Shakespeare.txt', 'r').read().lower()
filtered_lines = []
for line in poems.split('\n'):
    line = re.sub(r'[!@#$%^*\(\)\[\];:\r]', ' ', line)
    line = re.sub(r'[\s]+', ' ', line)
    if line != '':
        filtered_lines.append(line)
filtered_lines[:20]

X = []
Y = []
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_lines)
words = len(tokenizer.word_index) + 1
for line in filtered_lines:
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        X.append([token for token in tokens[:i]])
        Y.append([tokens[i]])

#Longest Sequence
longest_seq_len = max([len(seq) for seq in filtered_lines])
longest_seq_len    

X = np.array(pad_sequences(X, maxlen=longest_seq_len, padding='pre'))

Y = k_utils.to_categorical(Y, num_classes=words)

#Model
model = Sequential()

model.add(Embedding(words, 10, input_length=longest_seq_len))
model.add(LSTM(100, activation='tanh'))
model.add(Dropout(0.2))

model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='cosine_proximity', metrics=['accuracy'])

model.summary()

model.fit(X, Y, epochs=100)
    
    
    
from keras.models import save_model

save_model(model, 'poems_model_local_new.h5py')

import json
data_dict = {}
data_dict['word_ind_dic'] = tokenizer.word_index
data_dict['long_seq'] = longest_seq_len
open('poem_data_dict.json', 'w').write(json.dumps(data_dict))
"""
from keras.models import load_model
trained = load_model('S:/SUNY/Semester 4/Project/Capstone/Capstone/poems_model_local.h5py')

inp = "the black night sky beautiful and moon in the sky so bright"
for i in range(20):
    tokens = tokenizer.texts_to_sequences([inp])[0]
    tokens = np.array(pad_sequences([tokens], maxlen=longest_seq_len, padding='pre'))
    predicted = trained.predict_classes(tokens)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            inp += " " + word
            break
print(inp)
 """       
    
    
    
    
    
    
    