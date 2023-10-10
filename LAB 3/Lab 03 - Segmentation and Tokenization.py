# -*- coding: utf-8 -*-
"""Lab 03 - Segmentation and Tokenization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17EONomfrKBl7X48jSORE9GN2R5MMiTIA

## Sentence Segmentation
"""

from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

sampleText = "This programme is designed to provide students with knowledge and applied skills in data science, big data analytics and business intelligence. It aims to develop analytical and investigative knowledge and skills using data science tools and techniques, and to enhance data science knowledge and critical interpretation skills. Students will understand the impact of data science upon modern processes and businesses, be able to identify, and implement specific tools, practices, features and techniques to enhance the analysis of data."

Sentences = sent_tokenize(sampleText)

print("There are ", len(Sentences), "sentences in this text\n")

counter = 0
for sent in Sentences:
    counter+=1
    print(counter,".",sent,"\n")

"""## Word tokenization"""

from nltk.tokenize import word_tokenize

sampleText = "This programme is designed to provide students with knowledge and applied skills in data science, big data analytics and business intelligence. It aims to develop analytical and investigative knowledge and skills using data science tools and techniques, and to enhance data science knowledge and critical interpretation skills. Students will understand the impact of data science upon modern processes and businesses, be able to identify, and implement specific tools, practices, features and techniques to enhance the analysis of data."

Tokens = word_tokenize(sampleText)

print("There are ", len(Tokens), "tokens in this text\n")

counter = 0
for w in Tokens:
    counter+=1
    print(counter,".",w)
    
# To print all the tokens
print(Tokens)  
print()

# Use nltk.Text() to create a text list from the tokens list.
import nltk    
Tokenstext = nltk.Text(Tokens)
print(Tokenstext[0:len(Tokenstext)])

"""### Word Tokenisation"""

import nltk
from textblob import TextBlob
'''TextBlob - Installing/Upgrading 
From the PyPI
pip install -U textblob
From Conda
conda install -c conda-forge textblob'''

text = "Clairson International Corp. said it expects to report a net loss for its second quarter ended March 26 and doesn’t expect to meet analysts’ profit estimates of $3.9 to $4 million, or 76 cents a share to 79 cents a share, for its year ending Sept. 24."
wordtokens = text.split(' ')
print("Tokenisation using split function ~~~~~~~~~~~~~~~ ")
print(wordtokens)

print("\nTokenisation using nltk word tokenise function ~~~~~~~~~~~~~~~ ")
print(nltk.tokenize.word_tokenize(text))

print("\nTokenisation using textblob tokenise function ~~~~~~~~~~~~~~~ ")
print(TextBlob(text).words)

import spacy
nlp = spacy.load("en_core_web_sm")

text = "Clairson International Corp. said it expects to report a net loss for its second quarter ended March 26 and doesn’t expect to meet analysts’ profit estimates of $3.9 to $4 million, or 76 cents a share to 79 cents a share, for its year ending Sept. 24."

doc = nlp(text)
for token in doc:
    print(token.text)

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()

# Creating a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)
text = "Clairson International Corp. said it expects to report a net loss for its second quarter ended March 26 and doesn’t expect to meet analysts’ profit estimates of $3.9 to $4 million, or 76 cents a share to 79 cents a share, for its year ending Sept. 24."

tokens = tokenizer(text)
for token in tokens:
    print(token)

"""## Removing Stop Words & punctuation"""

import nltk, string
nltk.download('punkt')
nltk.download('stopwords')
import wordcloud
'''conda install -c conda-forge wordcloud
pip install wordcloud'''

import matplotlib.pyplot as plt

text = "This programme is designed to provide students with knowledge and applied skills in data science, big data analytics and business intelligence. It aims to develop analytical and investigative knowledge and skills using data science tools and techniques, and to enhance data science knowledge and critical interpretation skills. Students will understand the impact of data science upon modern processes and businesses, be able to identify, and implement specific tools, practices, features and techniques to enhance the analysis of data."
text_lower = text.lower()
wordtokens = nltk.tokenize.word_tokenize(text_lower)
print(wordtokens)

print("\nTotal number of words in this text corpus is ",len(wordtokens))

stopTokens = nltk.corpus.stopwords.words("english") + list(string.punctuation)
filteredTokens = []

for w in wordtokens:
    if w not in stopTokens :
        filteredTokens.append(w)

print("There are", len(filteredTokens), "words in this text after removing stop words\n")
print(filteredTokens)

print("\n", stopTokens)

# word count / bag of words
print(wordtokens.count("skills"))

fdist1 = nltk.FreqDist(filteredTokens)
print("\n",fdist1.most_common(10))

# Display the generated word cloud ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
word_cloud = wordcloud.WordCloud(background_color='white').generate(text) #untokenized text
plt.figure(figsize = (15, 8), facecolor = None) 
plt.imshow(word_cloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()