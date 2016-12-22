import json
import pandas as pd
import re, nltk
import numpy as np
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.metrics import edit_distance
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
 


tweets_data_path = '/home/candace/Documents/sentiment_analysis_dataset.csv'
classifier_NaiveBayes_path = '/home/candace/Documents/NaiveBayes.pickle'
vectorizer_CountVectorizer_path = '/home/candace/Documents/CountVectorizer.pickle'

whole_data_df = pd.read_csv(tweets_data_path, error_bad_lines=False)
train_data_df = whole_data_df[['Sentiment','SentimentText']]

# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
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
vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = stop_words,
    max_features = 100
)
 #fit the model and learns the vocabulary;
 #transform our corpus data into feature vectors.
corpus_data_features = vectorizer.fit_transform(
    train_data_df.SentimentText.tolist())

# Transfrom count to term frequency
tfidf_transformer = TfidfTransformer()
corpus_data_tfidf = tfidf_transformer.fit_transform(corpus_data_features)
#corpus_data_tfidf.shape

#Train the classifier
NB_model= MultinomialNB().fit(corpus_data_tfidf,  train_data_df.Sentiment)

#Save NaiveBayes classifier and vectorizer
clf_NB = open(classifier_NaiveBayes_path, 'wb')
#vec_cv = open(vectorizer_CountVectorizer_path,'wb')
pickle.dump(NB_model, clf_NB)
#pickle.dump(vectorizer,vec_cv)
clf_NB.close()
#vec_cv.close()
