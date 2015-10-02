from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
tfidf_vect= TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False, ngram_range=(2,2))

import pandas as pd
df = pd.read_csv('/Users/user/Desktop/CORPORA_EXPERIMENTO/corpora_listos/fixed_corpus_sust_adj.csv',
                     header=0, sep=',', names=['SentenceId', 'Sentence', 'Sentiment'])



reduced_data = tfidf_vect.fit_transform(df['Sentence'].values)
y = df['Sentiment'].values



from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(reduced_data,
                                                    y, test_size=0.33)
from sklearn.svm import SVC
#first svm
# print reduced_data.shape
# print y.shape
# print X_test.shape
# print X_train.shape
# print y_train
# print y_test

clf = SVC()
clf.fit(reduced_data, y)
prediction = clf.predict(X_test)
w = clf.coef_[0]
print w.toarray()
a = -w[0] / w[1]
xx = np.linspace(-10, 10)
yy = a * xx - clf.intercept_[0] / w[1]



# trying to get the separating hyperplane using weighted classes

#second svm
wclf = SVC(kernel='linear', class_weight={5: 10},C=1)
wclf.fit(reduced_data, y)
weighted_prediction = wclf.predict(X_test)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]

# plot separating hyperplanes and samples
import matplotlib.pyplot as plt
h0 = plt.plot(xx, yy, 'k-', label='no weights')
h1 = plt.plot(xx, wyy, 'k--', label='with weights')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y, cmap=plt.cm.Paired)
plt.legend()

plt.axis('tight')
plt.show()

