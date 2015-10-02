import numpy as np
import ______ as pd
import sklearn.feature_extraction.text
import sklearn.metrics
import sklearn.naive_bayes


names = ['text', 'label']
#read in some text data
data = pd.read_table("movie_ratings.txt",sep="\t", names=names)


#split the data intro training and testing sets
#Default is 75% train
#After the split, we essentially have to arrays of arrays
train, test = sklearn\
    .cross_validation.train_test_split(data, train_size=0.7)

train_data, test_data = pd.DataFrame(train, columns=names), \
                        pd.DataFrame(test, columns=names)

#Vectorization is the process of converting all names into a binary
#vector of 0s and 1s such that the name is encoded as a set of on/off
# atributes

vectorizer = sklearn.feature_extraction.text\
    .CountVectorizer(stop_words='english')


train_matrix = vectorizer.fit_transform(train_data['text'])
test_matrix = vectorizer.transform(test_data['text'])

positive_cases_train = (train_data['label']== 'POS')
positive_cases_test = (train_data['label']== 'POS')

#Train the classifier
#
classifier = sklearn.naive_bayes.MultinomialNB()
classifier.fit(train, positive_cases_train)


#Predict sentiment for the test set. Also note, it is possible
# to store the model of the classifier to disk for later use.
# Simple joblib from scikit-learn

predicted_sentiment = classifier.predict(test_matrix)
predicted_probs = classifier.predict_proba(test_matrix)

#Now, calculate the diagnostics
accuracy = classifier.score(test_matrix, positive_cases_test)
precision, recall, f1, _ = sklearn.\
    metrics.precision_recall_fscore_support(positive_cases_test,
                                            predicted_sentiment)


print(" ")
print("Accuracy= ", accuracy)
print("precision = ", precision)
print("Recall = ",recall)
print("F1 score - ", f1)