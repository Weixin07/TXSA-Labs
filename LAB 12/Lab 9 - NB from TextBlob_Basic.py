from textblob.classifiers import NaiveBayesClassifier
train = [
('I love this sandwich.', 'pos'),
('this is an amazing place!', 'pos'),
('I feel very good about these beers.', 'pos'),
('this is my best work.', 'pos'),
("what an awesome view", 'pos'),
('I do not like this restaurant', 'neg'),
('I am tired of this stuff.', 'neg'),
("I can't deal with this", 'neg'),
('he is my sworn enemy!', 'neg'),
('my boss is horrible.', 'neg')
]

test = [
('the beer was good.', 'pos'),
('I do not enjoy my job', 'neg'),
("I ain't feeling dandy today.", 'neg'),
("I feel amazing!", 'pos'),
('Gary is a friend of mine.', 'pos'),
("I can't believe I'm doing this.", 'neg')
]

cl = NaiveBayesClassifier(train)
print("\nAccuracy: %0.2f" % cl.accuracy(test))
print("\nThe sentence 'This is an amazing library!' has been classified as\n", cl.classify("This is an amazing library!"))
print()
print(cl.show_informative_features(5))
