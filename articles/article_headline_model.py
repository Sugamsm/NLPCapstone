import os
import json
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils as k_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

articles = []
for file in os.listdir('articles/data/'):
    if 'Article' in file:
        #print('File', file)
        headlines = pd.read_csv(file)['headline'].values
        for headline in headlines:
            if headline != 'Unknown':
                articles.append(headline.lower())
                break
articles[:10]

all_words = []
for headline in articles:
    all_words.extend(list(headline.split(' ')))
all_words = sorted(set(all_words))
all_words[:5]

#dictionaries
w_to_n = {}
n_to_w = {}
for index, word in enumerate(all_words):
    n_to_w[index] = word
    w_to_n[word] = index

len(all_words)

#generating training dataset

X = []
Y = []

for sentence in articles:
    tokens = sentence.split(' ')
    for i in range(1, len(tokens)):
        if i != len(tokens) - 1:
            X.append([w_to_n[word] for word in tokens[:i]])
            Y.append([w_to_n[tokens[i]]])

print(X[:10])
Y[:10]

X = np.array(X)
Y = np.array(Y)

longest_sequence = max([len(s) for s in articles])
print(longest_sequence)

X = np.array(pad_sequences(X, maxlen=longest_sequence, padding='pre'))

X = X.reshape(X.shape[0], 1, longest_sequence)

X.shape

Y = k_utils.to_categorical(Y, num_classes=len(all_words))
Y.shape

model = Sequential()

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), activation='sigmoid', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='sigmoid', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(64, activation='sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='cosine_proximity', metrics=['accuracy'])

model.summary()

model.fit(X, Y, epochs=500)

from keras.models import save_model
save_model(model, 'article_headlines.h5py')

article_data_dict = {}
article_data_dict['word_ind_dict'] = w_to_n
article_data_dict['ind_word_dict'] = n_to_w
article_data_dict['long_seq'] = longest_sequence
open('article_data_dict.json', 'w').write(json.dumps(article_data_dict))

#prediction

inp = 'politics has'.lower()

for i in range(5):
    words = nltk.word_tokenize(inp)
    sequence = []
    for word in words:
        sequence.append(w_to_n[word])
        sequences = [sequence]
        sequences = np.array(pad_sequences(sequences, maxlen=longest_sequence, padding='pre'))
        sequences = sequences.reshape(1, 1, longest_sequence)
    inp += " " + n_to_w[model.predict_classes(sequences)[0]]
print(inp)
"""
PREDICTION TEST


from keras.models import load_model

model2 = load_model('article_headlines.h5py')

inp = 'politics has'.lower()

for i in range(5):
    words = nltk.word_tokenize(inp)
    sequence = []
    for word in words:
        sequence.append(w_to_n[word])
        sequences = [sequence]
        sequences = np.array(pad_sequences(sequences, maxlen=longest_sequence, padding='pre'))
        sequences = sequences.reshape(1, 1, longest_sequence)
    inp += " " + n_to_w[model2.predict_classes(sequences)[0]]
print(inp)

"""