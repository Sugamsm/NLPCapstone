from keras.preprocessing.text import Tokenizer


def getPrediction(tokenizer, model, inp):
    for i in range(30):
        words = tokenizer.texts_to_sequences([inp])[0]
        sequence = pad_sequences(
            [words], maxlen=max_length_sequence - 1, padding='pre')
        predicted = model2.predict_classes(sequence, verbose=0)
        for prediction, index in tokenizer.word_index.items():
            output_word = ""
            if index == predicted:
                inp += " " + prediction
                break
    return inp
