# -*- coding: utf-8 -*-
"""Lab 04 - Text Normalization_Stemming and Lemmatization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12KQ8nNXiNlbpkvM0s_mxgExkPnOdt1Tp

# Stem vs. Lemma vs. Lexeme
#### A lemma is a word that stands at the head of a definition in a dictionary. All the head words in a dictionary are lemmas.
#### A lexeme is a unit of meaning, and can be more than one word. A lexeme is the set of all forms that have the same meaning.
#### In computational linguistics, a stem is the part of the word that never changes even when different forms of the word are used.

## Stemmers --> PorterStemmer
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words1 = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]
example_words2 = ["List", "listed", "lists", "listing", "listings"]

for w in example_words1:
    print(ps.stem(w))

for w in example_words2:
    print(ps.stem(w))

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

new_text = """It is very important to be pythonly while you are pythoning
        with python. All pythoners have pythoned poorly at least once."""

words = word_tokenize(new_text)

print([ps.stem(w) for w in words])

"""## Stemmers --> LancasterStemmer"""

from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government.  Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)

porter = PorterStemmer()
lancaster = LancasterStemmer()

print([porter.stem(t) for t in tokens])
print("\n")
print([lancaster.stem(t) for t in tokens])

"""## Lemmatization --> WordNetLemmatizer"""

from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
print("rocks :", lemmatizer.lemmatize("rocks"))
print("\nproduced :", lemmatizer.lemmatize("produced", pos = 'v'))

ps = PorterStemmer()
print("\nStem of the word produced :", ps.stem("produced"))

print("\nbetter :", lemmatizer.lemmatize("better", pos ="a"))

print("\nwomen :", lemmatizer.lemmatize("women", pos ="n"))

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government.  Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)

wnl = WordNetLemmatizer()
print([wnl.lemmatize(t) for t in tokens])
print()

for t in tokens:
    print ("{0:20}{1:20}".format(t, wnl.lemmatize(t, pos="v")))
print()
example_words = ["List", "listed", "lists", "listing", "listings"]
print([wnl.lemmatize(w) for w in example_words])
print()
for words in example_words:
    print ("{0:20}{1:20}".format(words, wnl.lemmatize(words, pos="v")))

"""# Lemmatization using TextBlob"""

from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
nltk.download('wordnet')

sentence = TextBlob('DENNIS: Listen, strange women lying in ponds distributing swords is no basis for a system of government. Supreme executive power derives from a mandate from the masses, not from some farcical aquatic ceremony.')
tokens = sentence.words
print(tokens)
print()
print(tokens.lemmatize())
print()

for t in tokens:
    print ("{0:20}{1:20}".format(t, wnl.lemmatize(t, pos="v")))

from textblob import TextBlob
text = TextBlob("List listed lists listing listings")
tokens = text.words
print(tokens)
print("\n",tokens.lemmatize())
print()
for t in tokens:
    print ("{0:20}{1:20}".format(t, wnl.lemmatize(t, pos="v")))

"""# Lemmatization using SPACY"""

import spacy

nlp = spacy.load("en_core_web_sm")

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government.  Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""

doc = nlp(raw)

for token in doc:
    print(f"{token} -> {token.lemma_}" )

"""# Stemming & Lemmatization

### Stemming and Lemmatization both generate the root form of the inflected words. The difference is that stem might not be an actual word whereas, lemma is an actual language word.
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

#file = open ("D:/APU/TXSA-CT107-3-3/TUTORIAL/sample01.txt")
#raw = file.read()


raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government.  Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""

words = raw.lower()
print(words)
print()
tokens = word_tokenize(words)
print("Tokens")
print(tokens)
print()
print("Lemmas")
wnl = nltk.WordNetLemmatizer()
print([wnl.lemmatize(t, pos = "v") for t in tokens])
print()
print("Porter Stemming")
ps = PorterStemmer()
print ([ps.stem(t) for t in tokens])
print()
print("Lancaster Stemming")
ls = LancasterStemmer()
print ([ls.stem(t) for t in tokens])
print()
print("Snowball Stemming")
sn = nltk.SnowballStemmer("english")
print([sn.stem(t) for t in tokens])

"""# Stemmers --> SnowballStemmer"""

import nltk
nltk.download('punkt')

print(nltk.SnowballStemmer.languages)
print(len(nltk.SnowballStemmer.languages))
print()
text = "This is achieved in practice during stemming, a text preprocessing operation."
tokens = nltk.tokenize.word_tokenize(text)
print()
stemmer = nltk.SnowballStemmer('english')
print([stemmer.stem(t) for t in tokens])
print()
text2 = "Ceci est réalisé en pratique lors du stemming, une opération de prétraitement de texte."
tokens2 = nltk.tokenize.word_tokenize(text2)
print()
stemmer = nltk.SnowballStemmer('french')
print([stemmer.stem(t) for t in tokens2])

"""## SnowballStemmer --> for other space delimited languages

Lets see how to **detect and translate a language**

**langdetect** supports 55 languages out of the box (ISO 639-1 codes)

https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
"""

!pip install langdetect

from textblob import TextBlob
import nltk
nltk.download('punkt')
from langdetect import detect
text = "This is achieved in practice during stemming, a text preprocessing operation."
print(detect(text))

en_blob = TextBlob(text)
fr_blob = en_blob.translate(from_lang="en", to='fr')
print(fr_blob)

tokens = fr_blob.words
print(tokens)
print()

stemmer = nltk.SnowballStemmer('french')
print([stemmer.stem(t) for t in tokens])

# Language Detection and Translation
!pip install translate==3.6.1

# Language Detection and Translation
from langdetect import detect
text = "Ceci est réalisé en pratique lors du stemming, une opération de prétraitement de texte"
print(detect(text))

from translate import Translator
translator = Translator(from_lang = 'fr', to_lang='en')
print(translator.translate(text))