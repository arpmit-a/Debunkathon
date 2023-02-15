from preprocess import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

def vectorize_for_models(train_data,test_data):
    vectorization = TfidfVectorizer()
    train_data=vectorization.fit_transform(train_data)
    test_data=vectorization.transform(test_data)
    return train_data,test_data,vectorization

def vectorize_for_lstm(train_data,test_data,max_words,max_len):
    token = Tokenizer(num_words=max_words, lower=True, split=' ')
    token.fit_on_texts(x_train.values)
    sequences = token.texts_to_sequences(train_data.values)
    train_padded = pad_sequences(sequences, maxlen=max_len)
    test_sequences = token.texts_to_sequences(test_data)
    test_padded = pad_sequences(test_sequences,maxlen=max_len)
    return train_padded,test_padded