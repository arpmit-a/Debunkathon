import argparse
import model
import os
import preprocess
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import logging
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--title", help="Enter your title",default='')
    parser.add_argument("-i", "--text", help="Enter your news article text here",default='')
    parser.add_argument("-d", "--dataset", help="Enter your dataset link",default='')
    args= parser.parse_args()
    if os.path.exists("/content/model.pkl"):
        loaded_model = pickle.load(open("/content/model.pkl", "rb"))
        target=remove_punctuations(args.text)
        target=remove_stopwords(target)
        target=remove_url(target)
        target=clean_chars(target)
        target=remove_emojis(target)
        target=[target]
        vectorization = TfidfVectorizer()
        df=pd.read_csv(args.dataset)
        text=vectorization.fit_transform(df['text'])
        target=vectorization.transform(target)
        try:
            prediction=loaded_model.predict(target)
            logging.info(f"Your news is {prediction}")
        except:
            prediction=loaded_model.predict(target.todense())
            logging.info(f"Your news is {prediction}")

    else:
        logger.info("Please generate the .pkl file first by running model.py")
        