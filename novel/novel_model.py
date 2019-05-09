import re
import json
import numpy as np

from keras.preprocessing.sequence import pad_sequences
import keras.utils as k_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import nltk
#nltk.download('all-nltk')
import os
print(os.getcwd())

text = open('novel/TheReturnOfSherlockHolmes.txt', 'rb').read().decode(encoding='utf-8-sig').lower()
text = re.sub(r'[#$@\[\]\\/_\+=\(\):;<>\*\r+]', '', text)
text = re.sub(r'[\n+]', '', text)
text = re.sub(r'[\s+]', ' ', text)
sentences = nltk.sent_tokenize(text)
sentences = sentences[:666]

words = []
for sentence in sentences:
    words.extend(list(nltk.word_tokenize(sentence)))
words = sorted(set(words))
words_length = len(words)
print(words_length)

#dictionaries
w_to_n = {}
n_to_w = {}

w_to_n = {word:n for n, word in enumerate(words)}
n_to_w = {n:word for n, word in enumerate(words)}

#Datasets preparation

X = []
Y = []
for sentence in sentences:
    tokens = nltk.word_tokenize(sentence)
    for i in range(1, len(tokens)):
        X.append([w_to_n[token] for token in tokens[:i]])
        Y.append([w_to_n[tokens[i]]])

longest_sequence = max([len(i) for i in sentences])

X = np.array(pad_sequences(X, maxlen=longest_sequence, padding='pre'))
X = X.reshape(X.shape[0], 1, X.shape[-1])
Y = k_utils.to_categorical(Y, num_classes=words_length)
#X = X.reshape(X.shape[0], max_length_sequence - 1, 1)

model = Sequential()

model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='cosine_proximity', metrics=['accuracy'])

model.summary()

model.fit(X, Y, epochs=20)

model.save('novel_model.h5py')

#Saving dictionaries for later/offline use
data_dict = {}
data_dict['word_ind_dic'] = w_to_n
data_dict['ind_word_dic'] = n_to_w
data_dict['long_seq'] = longest_sequence
open('novel_data_dict.json', 'w').write(json.dumps(data_dict))

"""
inp = "he went there alone looking for a fall"

for i in range(30):
    words = nltk.word_tokenize(inp)
    sequence = pad_sequences([words], maxlen=longest_sequence, padding='pre')
    predicted = model.predict_classes(sequence, verbose=0)
    for word, index in w_to_n.items():
        if index == predicted:
            inp += " " + output_word
            break
print(inp)
"""