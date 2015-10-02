import pandas as pd

df = pd.read_csv('/Users/user/Desktop/MICAI_2/prueba_todos.csv',
                     header=0, sep=',', names=['id', 'content', 'label'])

#Feature 1
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(use_idf=True, smooth_idf=True,
                             sublinear_tf=False, ngram_range=(2,2))
#Feature 2
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer()

#Feature 3
from sklearn.feature_extraction.text import HashingVectorizer
hash_vect = HashingVectorizer()

#Feature 4
from gensim.models import word2vec
w2vec = word2vec()

from sklearn.pipeline import  FeatureUnion
combined_features = FeatureUnion([("tfidf_vect", tfidf_vect),
                                  ("bow", bow),
                                  ("hash",hash_vect),
                                  ('word2vec',word2vec)])


X_combined_features = combined_features.fit_transform(df['content'].values)
y = df['label'].values

print X_combined_features.toarray()




from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_combined_features,
                                                    y, test_size=0.33)

################################################
###################### balanceado ##############
################################################
from sklearn.svm import SVC
wclf = SVC(kernel='linear', C= 1, class_weight={1: 10})
wclf.fit(X_combined_features, y)
weighted_prediction = wclf.predict(X_test)


from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

print 'Balanceado:\n'
print 'Accuracy:', accuracy_score(y_test, weighted_prediction)
print 'F1 score:', f1_score(y_test, weighted_prediction,average='weighted')
print 'Recall:', recall_score(y_test, weighted_prediction,
                              average='weighted')
print 'Precision:', precision_score(y_test, weighted_prediction,
                                    average='weighted')
print '\n clasification report:\n', classification_report(y_test, weighted_prediction)
print '\n confussion matrix:\n',confusion_matrix(y_test, weighted_prediction)

##########################################################
########################balanceo automatico###############
##########################################################

from sklearn.svm import SVC
auto_wclf = SVC(kernel='linear', C= 1, class_weight='auto')
auto_wclf.fit(X_combined_features, y)
auto_weighted_prediction = auto_wclf.predict(X_test)


from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

print 'balanceo automatico:\n'

print 'Accuracy:', accuracy_score(y_test, auto_weighted_prediction)

print 'F1 score:', f1_score(y_test, auto_weighted_prediction,
                            average='weighted')

print 'Recall:', recall_score(y_test, auto_weighted_prediction,
                              average='weighted')

print 'Precision:', precision_score(y_test, auto_weighted_prediction,
                                    average='weighted')

print '\n clasification report:\n', classification_report(y_test,auto_weighted_prediction)

print '\n confussion matrix:\n',confusion_matrix(y_test, auto_weighted_prediction)

#####################################################
######################## sin balanceo ###############
#####################################################

from sklearn.svm import SVC
clf = SVC(kernel='linear', C= 1)
clf.fit(X_combined_features, y)
prediction = clf.predict(X_test)


from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from sklearn.metrics import precision_recall_fscore_support as score

print 'Sin balanceo:\n'


print 'F1 score:', f1_score(y_test, prediction, average='weighted')
print 'Accuracy:', accuracy_score(y_test, prediction)
print 'Recall:', recall_score(y_test, prediction,average='weighted')
print 'Precision:', precision_score(y_test, prediction,average='weighted')
print '\n clasification report:\n', classification_report(y_test,prediction)
print '\n confussion matrix:\n',confusion_matrix(y_test, prediction)
