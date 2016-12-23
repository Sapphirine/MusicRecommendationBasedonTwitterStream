import os
import json
import pandas as pd
import re, nltk
import numpy as np
import pickle
import enchant
import random
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer    
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC  

#set work directory
os.chdir("/home/candace/Documents/sorted_tweets")

sorted_tweets_path = '/home/candace/Documents/sorted_tweets'
song_list_path = '/home/candace/Documents/song_list.csv'
vectorizer_CountVectorizer_path = '/home/candace/Documents/CountVectorizer.pickle'
vectorizer_TfidfVectorizer_path = '/home/candace/Documents/TfidfVectorizer.pickle'
classifier_LogisticRegression_path = '/home/candace/Documents/LogisticRegression.pickle'
classifier_NaiveBayes_path = '/home/candace/Documents/NaiveBayes.pickle'
classifier_LinearSVC_path = '/home/candace/Documents/LinearSVC.pickle'

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
    
dt = enchant.Dict("en")
max_dist = 2
def sp_correct(word):
    if dt.check(word):
        return word
    suggestions = dt.suggest(word)
    if suggestions and edit_distance(word, suggestions[0]) <= max_dist:
        return suggestions[0]
    else:
        return word

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # correct spell
    words = sp_correct(tokens)
    # stem
    stems = stem_tokens(words, stemmer)
    return stems

stop_words = stopwords.words('english')
# remove http,rt etc
stop_words.append(u'http')
stop_words.append(u'rt')
stop_words.append(u'www')
stop_words.append(u'com')
stop_words.append(u'quot')
######## 

#Load vectorizers and classifiers
vec_cv_f = open(vectorizer_CountVectorizer_path)
vec_cv = pickle.load(vec_cv_f)
vec_cv_f.close()

vec_tfidf_f = open(vectorizer_TfidfVectorizer_path)
vec_tfidf = pickle.load(vec_tfidf_f)
vec_tfidf_f.close()

clf_LG_f = open(classifier_LogisticRegression_path)
clf_LG = pickle.load(clf_LG_f)
clf_LG_f.close()

clf_NB_f = open(classifier_NaiveBayes_path)
clf_NB = pickle.load(clf_NB_f)
clf_NB_f.close()

clf_SVC_f = open(classifier_LinearSVC_path)
clf_SVC = pickle.load(clf_SVC_f)
clf_SVC_f.close()

#Read song list
with open(song_list_path) as f:
    song_list = f.read().splitlines()

for i in range(len(song_list)):
    #Load sorted tweets
    pred_tweets = pd.read_csv(filepath_or_buffer = str(i),
                              names = ['index','text'])
    #pass empty file                          
    if len(pred_tweets) == 0:
        print song_list[i],'empty file'
#        print song_list[i],',','nan'
        
    else:
        #extract feature                           
        features_cv = vec_cv.fit_transform(pred_tweets.text.tolist())
        features_tfidf = vec_tfidf.fit_transform(pred_tweets.text.tolist())
    
        if np.shape(features_cv)[1] >= 100:
            features_nd_cv = features_cv.toarray()
            features_nd_tfidf = features_tfidf.toarray()
            
            #predict sentiment
            text_pred_LG = clf_LG.predict(features_nd_cv)
            text_pred_NB = clf_NB.predict(features_tfidf)
            text_pred_SVC = clf_SVC.predict(features_nd_tfidf)
    
            #store result
            pred_tweets['pred_sent_LG'] = text_pred_LG.tolist()
            pred_tweets['pred_sent_NB'] = text_pred_NB.tolist()
            pred_tweets['pred_sent_SVC'] = text_pred_SVC.tolist()
            pred_tweets.to_csv(path_or_buf = "pred %s"%str(i),encoding = 'utf-8')
    
            #calculate rate of different model
            neg = 0
            pos = 0
            for j in range(len(text_pred_LG.tolist())):
                if text_pred_LG.tolist()[j] == 0:
                    neg = neg+1
                else:
                    pos = pos+1
            rate = 5.0*(pos)/(pos+neg)
            print song_list[i],"model:Logistic Regression",'\n',"neg:",neg, "pos:",pos,"\n","rate:",rate
#            print song_list[i],',',neg,',', pos,',',rate
            neg = 0
            pos = 0
            for j in range(len(text_pred_NB.tolist())):
                if text_pred_NB.tolist()[j] == 0:
                    neg = neg+1
                else:
                    pos = pos+1
            rate = 5.0*(pos)/(pos+neg)
            print "model:Navie Bayes",'\n',"neg:",neg, "pos:",pos,"\n","rate:",rate
#            print song_list[i],',',neg,',', pos,',',rate
            neg = 0
            pos = 0
            for j in range(len(text_pred_SVC.tolist())):
                if text_pred_SVC.tolist()[j] == 0:
                    neg = neg+1
                else:
                    pos = pos+1
            rate = 5.0*(pos)/(pos+neg)
            print "model:Linear SVM",'\n',"neg:",neg, "pos:",pos,"\n","rate:",rate
#            print song_list[i],',',neg,',', pos,',',rate
    
        else:
            print song_list[i],"feature number:", np.shape(features_cv)[1], "is not enough"
#            print song_list[i],",","nan"
            pass
