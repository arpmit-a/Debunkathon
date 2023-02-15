import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn import preprocessing
nltk.download('wordnet')
nltk.download('omw-1.4')
ps = PorterStemmer()
wordnet_map = {"N":wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
def get_data(df):
    cols=df.columns
    for i in cols:
        if i=='text' or i=='label':
            continue
        else:
            df=df.drop(i,axis=1)
    return df

def lower_text(df):
    df['text'] = df['text'].str.lower()
    return df

def remove_punctuations(text):
    res = re.sub(r'[^\w\s]', '', text)
    return res

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopwords.words('english')])


def find_freq_words(df):
    word_count = Counter()
    for text in df['text']:
        for word in text.split():
            word_count[word] += 1

    FREQUENT_WORDS = set(word for (word, wc) in word_count.most_common(3))
    return FREQUENT_WORDS

def remove_freq_words(text,FREQUENT_WORDS):
    return " ".join([word for word in text.split() if word not in FREQUENT_WORDS])

def stemming(text):
    return " ".join([ps.stem(word) for word in text.split()])

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def clean_chars(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)



def preprocess(csv_file):
    df=pd.read_csv(csv_file)
    df=get_data(df)
    df=lower_text(df)
    df['text']=df['text'].apply(lambda x : remove_punctuations(x))
    print("punctuations removed")
    df['text']=df['text'].apply(lambda x: remove_stopwords(x))
    print("stopwords removed")
    FREQUENT_WORDS=find_freq_words(df)
    df['text']=df['text'].apply(lambda x: remove_freq_words(x,FREQUENT_WORDS))
    print("repeated words removed")
    df['text']=df['text'].apply(lambda x: stemming(x))
    print("stemming done")
    df['text']=df['text'].apply(lambda x: lemmatize_words(x))
    print("lemmetization done")
    df['text']=df['text'].apply(lambda x: remove_url(x))
    print("URLS removed")
    df['text']=df['text'].apply(lambda x: clean_chars(x))
    print("Characters removed")
    df['text']=df['text'].apply(lambda x: remove_emojis(x))
    print("emojis removed")
    le = preprocessing.LabelEncoder()
    df['label']=le.fit_transform(df['label'])
    return df



    

    
