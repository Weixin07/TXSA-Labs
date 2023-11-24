import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow

# Load the dataset
data = pd.read_csv('D:/APU/TXSA-CT107-3-3/LAB/LAB 8/opinion_dataset.csv')
data.head(n=20)
data['Sentiment'].value_counts()

# Target variable visualisation
Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Opinions'], color=('r','g'))
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()

# Plot the target variable using seaborn
import seaborn as sns
sns.countplot(data['Sentiment'], palette="Set2")


X = data['Opinions'].values
X
y = data['Sentiment'].values
y

# Count Vectorization
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range = (1,1), tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Opinions'])


# Data split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['Sentiment'], test_size=0.3, random_state=1)


# MultinomialNB fomr Sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
MNB_classifier = MultinomialNB().fit(X_train, y_train)
predicted_MNB = MNB_classifier.predict(X_test)
print("Accuracy of MNB Classifier:", metrics.accuracy_score(y_test, predicted_MNB))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_MNB))

sensitivity_specificity_support(y_true, y_pred, average='macro')

# Logistic Regression form Sklearn
from sklearn.linear_model import LogisticRegression
LR_classifier = LogisticRegression().fit(X_train, y_train)
predicted_LR= LR_classifier.predict(X_test)
print("Accuracy of LR Classifier:", metrics.accuracy_score(y_test, predicted_LR))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_LR))


# SGD Classifier form Sklearn
from sklearn.linear_model import SGDClassifier
SGD_classifier = SGDClassifier().fit(X_train, y_train)
predicted_SGD= SGD_classifier.predict(X_test)
print("Accuracy of SGD Classifier:", metrics.accuracy_score(y_test, predicted_SGD))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_SGD))


# SVC Classifier from Sklearn
from sklearn.svm import SVC
SVC_classifier = SVC().fit(X_train, y_train)
predicted_SVC= SVC_classifier.predict(X_test)
print("Accuracy of SVC Classifier:", metrics.accuracy_score(y_test, predicted_SVC))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_SVC))


# LinearSVC Classifier from Sklearn
from sklearn.svm import LinearSVC
LSVC_classifier = LinearSVC().fit(X_train, y_train)
predicted_LSVC= LSVC_classifier.predict(X_test)
print("Accuracy of LinearSVC Classifier:", metrics.accuracy_score(y_test, predicted_LSVC))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_LSVC))


# Decsion Tree Classifier from Sklearn
from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier().fit(X_train, y_train)
predicted_DT= DT_classifier.predict(X_test)
print("Accuracy of DT Classifier:", metrics.accuracy_score(y_test, predicted_DT))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_DT))


# Random Forest Classifier from Sklearn
from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier().fit(X_train, y_train)
predicted_RF = RF_classifier.predict(X_test)
print("Accuracy of RF Classifier:", metrics.accuracy_score(y_test, predicted_RF))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_RF))


# AdaBoost Classifier from Sklearn
from sklearn.ensemble import AdaBoostClassifier
ADB_classifier = AdaBoostClassifier().fit(X_train, y_train)
predicted_ADB = ADB_classifier.predict(X_test)
print("Accuracy of ADB Classifier:", metrics.accuracy_score(y_test, predicted_ADB))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_ADB))


# KNN Classifier from Sklearn
from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier().fit(X_train, y_train)
predicted_KNN = KNN_classifier.predict(X_test)
print("Accuracy of KNN Classifier:", metrics.accuracy_score(y_test, predicted_KNN))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_KNN))


# Multi Layer Perceptron Classifier from Sklearn
from sklearn.neural_network import MLPClassifier
MLP_classifier = MLPClassifier(max_iter=1000).fit(X_train, y_train)
predicted_MLP = MLP_classifier.predict(X_test)
print("Accuracy of MLP Classifier:", metrics.accuracy_score(y_test, predicted_MLP))
print("Classification Report:\n", metrics.classification_report(y_test, predicted_MLP))


# Confusion Matrix for MLP
from sklearn.metrics import confusion_matrix
CM_MLP = confusion_matrix(y_test, predicted_MLP)
CM_MLP


# ~~~~~~~~~~~~~~~~~~~~~~~ Model Comparison ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("~~~~~~~~~~~~~~ Accuracies of the Classifiers ~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\nAccuracy of MNB Classifier:", metrics.accuracy_score(y_test, predicted_MNB))
print("\nAccuracy of LR Classifier:", metrics.accuracy_score(y_test, predicted_LR))
print("\nAccuracy of SGD Classifier:", metrics.accuracy_score(y_test, predicted_SGD))
print("\nAccuracy of SVC Classifier:", metrics.accuracy_score(y_test, predicted_SVC))
print("\nAccuracy of LinearSVC Classifier:", metrics.accuracy_score(y_test, predicted_LSVC))
print("\nAccuracy of DT Classifier:", metrics.accuracy_score(y_test, predicted_DT))
print("\nAccuracy of RF Classifier:", metrics.accuracy_score(y_test, predicted_RF))
print("\nAccuracy of ADB Classifier:", metrics.accuracy_score(y_test, predicted_ADB))
print("\nAccuracy of KNN Classifier:", metrics.accuracy_score(y_test, predicted_KNN))
print("\nAccuracy of MLP Classifier:", metrics.accuracy_score(y_test, predicted_MLP))


# Save and Load the classifier
import pickle
print(" \n~~~~~~~~~ Save and Load the classifier  ~~~~~~~~~ ")
saved_model = pickle.dumps(SVC_classifier) 
SVC_from_pickle = pickle.loads(saved_model) 
SVC_from_pickle.predict(X_test) 

