import json
import pandas as pd
import re, nltk
import numpy as np
import pickle
import enchant
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.metrics import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer   
from sklearn.svm import LinearSVC   


tweets_data_path = '/home/candace/Documents/sentiment_analysis_dataset.csv'
classifier_LinearSVC_path = '/home/candace/Documents/LinearSVC.pickle'
vectorizer_CountVectorizer_path = '/home/candace/Documents/CountVectorizer.pickle'
vectorizer_TfidfVectorizer_path = '/home/candace/Documents/TfidfVectorizer.pickle'

whole_data_df = pd.read_csv(tweets_data_path, error_bad_lines=False)
train_data_df = whole_data_df[['Sentiment','SentimentText']]

# based on http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
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


##Defining a vectorizer for feature extraction
#vectorizer = CountVectorizer(
#    analyzer = 'word',
#    tokenizer = tokenize,
#    lowercase = True,
#    stop_words = stop_words,
#    max_features = 100
#)

#equals to cv+transform
vectorizer = TfidfVectorizer(
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
    
#convert the result to an array
corpus_data_features_nd = corpus_data_features.toarray()
#corpus_data_features_nd.shape

## Transfrom count to term frequency
#tfidf_transformer = TfidfTransformer()
#corpus_data_tfidf = tfidf_transformer.fit_transform(corpus_data_features)
##corpus_data_tfidf.shape

#Train the classifier
SVC_model= LinearSVC(C=0.5).fit(corpus_data_features_nd, train_data_df.Sentiment)

#Save NaiveBayes classifier and vectorizer
clf_SVC = open(classifier_LinearSVC_path, 'wb')
#vec_cv = open(vectorizer_CountVectorizer_path,'wb')
vec_Tfidf = open(vectorizer_TfidfVectorizer_path,'wb')
pickle.dump(SVC_model, clf_SVC)
#pickle.dump(vectorizer,vec_cv)
pickle.dump(vectorizer,vec_Tfidf)
clf_SVC.close()
#vec_cv.close()
vec_Tfidf.close()
