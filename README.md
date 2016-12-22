# MusicRecommendationBasedonTwitterStream

We set up a music recommender website using twitter streaming data and explore the sentimental analysis to decide the recommended items.
The twitter test dataset was extracted by some music keywords from twitter streaming API. The train dataset was from the website sentiment140. We use the NLTK to preprocess the raw data and countvectorizer, tfidfvectorizer to extracted feature. Naïve Bayes, Logistic regression and linear SVM were chosen to train the classifier model by mahout and python. Evaluate three models and test the best one through twitter streaming data. Finally show the recommendation result by the website which is formed through javascript.

# Software Package Description
Py_train_clf: Contains python script for data processing supported by NLTK and enchant package and training vectorizers and classifiers using scikit-learn package;
Mh_clf: Contains java script for data processing using chimpler jar and mahout script for classfier training; 
Py_clf_evl: Contains various evaluation methods in python script for classifiers;
Streaming _tw_API: Contains python scripts of extract streaming twitter with tweepy package
Sort_tw: For sorting streaming twitter data
Pred_tw: Contains python scripts for sentiment prediction on twitter dataset and music rating system.
Web_rec: Contains java script and html to establish the website page.
