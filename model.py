from keras.layers import *
from keras.models import *
import keras.backend as K
from keras.layers import *
from keras.models import *
import keras.backend as K #for some advanced functions 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from vectorizer import vectorize_for_models
from vectorizer import vectorize_for_lstm
import preprocess
import argparse
def make_dataset_for_models(df):
        x=df.text
        y=df.label
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 13)
        x_train,x_test= vectorize_for_models(x_train,x_test)[0],vectorize_for_models(x_train,x_test)[1]
        return x_train,x_test,y_train,y_test
def make_dataset_for_lstm(df):
        x=df.text
        y=df.label
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 13)
        x_train,x_test= vectorize_for_lstm(x_train,x_test)
        return x_train,x_test,y_train,y_test

def PassiveAggressive(df):
        x_train,x_test,y_train,y_test=make_dataset_for_models(df)
        model = PassiveAggressiveClassifier(C = 0.5, random_state = 5)
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        return accuracy_score(y_test, test_pred),model

def DecisionTree(df):
        x_train,x_test,y_train,y_test=make_dataset_for_models(df)
        model = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=5, min_samples_leaf=2)
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        return accuracy_score(y_test, test_pred),model

def RandomForest(df):
        x_train,x_test,y_train,y_test=make_dataset_for_models(df)
        model = RandomForestClassifier(n_estimators = 100)
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        return accuracy_score(y_test, test_pred),model

def naivebayes(df):
        x_train,x_test,y_train,y_test=make_dataset_for_models(df)
        x_train=x_train.todense()
        x_test=x_test.todense()
        classifier = GaussianNB() 
        classifier.fit(x_train, y_train)  
        test_pred = model.predict(x_test)
        return accuracy_score(y_test, test_pred),model

def svm(df):
        x_train,x_test,y_train,y_test=make_dataset_for_models(df)
        model = GDClassifier(loss="hinge", penalty="l2")
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        return accuracy_score(y_test, test_pred),model

def lstm(embed_dim ,lstm_out ,batch_size,max_words,max_len):
        model = Sequential()
        model.add(Embedding(max_words, embed_dim, input_length = max_len))
        model.add(LSTM(lstm_out))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, name='out_layer'))
        model.add(Activation('sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer='adam',
               metrics = ['accuracy'])
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--dataset", help="Enter your dataset path",default='')
    args = parser.parse_args()
    df=preprocess(args.dataset)
    acc=[]
    acc.append(PassiveAggressive(df)[0])
    acc.append(DecisionTree(df)[0])
    acc.append(RandomForest(df)[0])
    acc.append(naivebayes(df)[0])
    acc.append(svm(df)[0])
    if acc.index(max(acc))==0:
        model=PassiveAggressive(df)[1]
        pickle.dump(model, open("/content/model.pkl", "wb"))
    elif acc.index(max(acc))==1:
        model=DecisionTree(df)[1]
        pickle.dump(model, open("/content/model.pkl", "wb"))
    elif acc.index(max(acc))==2:
        model=RandomForest(df)[1]
        pickle.dump(model, open("/content/model.pkl", "wb"))
    elif acc.index(max(acc))==3:
        model=naivebayes(df)[1]
        pickle.dump(model, open("/content/model.pkl", "wb"))
    elif acc.index(max(acc))==4:
        model=svm(df)[1]
        pickle.dump(model, open("/content/model.pkl", "wb"))
    embed_dim = 100
    lstm_out = 64
    batch_size = 32
    max_words = 2000
    max_len = 400
    model=lstm(embed_dim ,lstm_out ,batch_size,max_words,max_len)
    x_train,x_test,y_train,y_test=make_dataset_for_lstm(df)
    model.fit(x_train, y_train, batch_size=batch_size, epochs = 3, validation_split=0.2)
    scores = model.evaluate(x_test,y_test, verbose=0)
    if scores[1]>max(acc):
        model_json=model.to_json()
        with open("/content/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("/content/model.h5")




