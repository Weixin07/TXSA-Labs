import nltk
from nltk.corpus import names
import random

def gender_features(word):
    return {'suffix1': word[-1:],
            'suffix2': word[-2:]}
gender_features("Mafas")

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
[(name, 'female') for name in names.words('female.txt')])

random.shuffle(labeled_names)

train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]

train_set = [(gender_features(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print("\nThe validation accuracy is: ", nltk.classify.accuracy(classifier, devtest_set)*100,"%")
print("\nThe testing accuracy is: ", nltk.classify.accuracy(classifier, test_set)*100,"%")

errors = []
prediction_list = []
TP=0
TN=0
FP=0
FN=0
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    prediction_list.append((tag, guess, name))
    if guess == 'male':
        if tag == 'male':
            TP+=1
        else:
            FP+=1
    else:
        if tag == 'female':
            TN+=1
        else:
            FN+=1       
  
    if guess != tag:
        errors.append((tag, guess, name))

print('\nTotal number of test items: ',len(prediction_list))
print('Total number of errors: ', len(errors))

print("\n~~~~~~ Classification ~~~~~~" )
print('TP: ', TP)
print('FP: ', FP)
print('TN: ', TN)
print('FN: ', FN)

def column(matrix, i):
    return [row[i] for row in matrix]

print("\n~~~~~~ Confusion Matrix ~~~~~~")
cm = nltk.ConfusionMatrix(column(prediction_list,0), column(prediction_list,1))
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
