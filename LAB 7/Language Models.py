#!/usr/bin/env python
# coding: utf-8

# ### Updating and checking the NLTK version

# In[1]:


# !pip install -U pip
# !pip install -U dill
# !pip install -U nltk==3.4


# In[2]:


import nltk
print(nltk.__version__)


# # N-gram using NLTK

# Traditionally, we can use n-grams to generate language models to predict which word comes next given a history of words.
# 
# We'll use the lm module in nltk to get a sense of how non-neural language modelling is done.

# In[3]:


from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten


# If we want to train a bigram model, we need to turn this text into bigrams. Here's what the first sentence of our text would look like if we use the ngrams function from NLTK for this.

# In[4]:


text = "I am learning Text Analytics"
tokens = nltk.tokenize.word_tokenize(text.lower())
list(bigrams(tokens))


# In[5]:


list(ngrams(tokens, n=3))


# Add special "padding" symbols to the sentence before splitting it into ngrams. Fortunately, NLTK also has a function for that, let's see what it does to the first sentence.

# In[6]:


from nltk.util import pad_sequence
list(pad_sequence(tokens, pad_left=True, left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>", n=2)) 
# The n order of n-grams, if it's 2-grams, you pad once, 3-grams pad twice, etc. 


# In[7]:


padded_sent = list(pad_sequence(tokens, pad_left=True, left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>", n=2))
list(ngrams(padded_sent, n=2))


# Note the n argument, that tells the function we need padding for bigrams.
# 
# Now, passing all these parameters every time is tedious and in most cases they can be safely assumed as defaults anyway.
# 
# Thus the nltk.lm module provides a convenience function that has all these arguments already set while the other arguments remain the same as for pad_sequence.

# In[8]:


from nltk.lm.preprocessing import pad_both_ends
list(pad_both_ends(tokens, n=2))


# Combining the two parts discussed so far we get the following preparation steps for one sentence.

# In[9]:


list(bigrams(pad_both_ends(tokens, n=2)))


# To make our model more robust we could also train it on unigrams (single words) as well as bigrams, its main source of information. NLTK once again helpfully provides a function called everygrams.
# 
# While not the most efficient, it is conceptually simple.

# In[10]:


from nltk.util import everygrams
padded_bigrams = list(pad_both_ends(tokens, n=2))
list(everygrams(padded_bigrams, max_len=1))


# In[11]:


list(everygrams(padded_bigrams, max_len=2))


# During training and evaluation our model will rely on a vocabulary that defines which words are "known" to the model.
# 
# To create this vocabulary we need to pad our sentences (just like for counting ngrams) and then combine the sentences into one flat stream of words.

# ### Calculating probability of n-grams in a text of sentences

# In[12]:


import nltk
text = "I am learning Text Analytics"
# Tokenize the text.
tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(text)))]
print(tokenized_text)


# In[13]:


# Preprocess the tokenized text for 3-grams language modelling
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

n = 3
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

model = MLE(n) # Lets train a 3-grams maximum likelihood estimation model.
model.fit(train_data, padded_sents)


# To get the counts:

# In[14]:


model.counts['i'] # i.e. Count('i')


# In[15]:


model.counts[['i']]['am'] # i.e. Count('am'|'i')


# In[16]:


model.counts[['i', 'am']]['learning'] # i.e. Count('learning'|'i am')


# In[17]:


model.score('am', 'i'.split())  # P('am'|'i')


# In[18]:


model.score('learning', 'i am'.split())  # P('learning'|'i am')


# ## N-gram using NLTK

# In[19]:


import nltk
from nltk.util import ngrams
 
# Function to generate n-grams from sentences.
def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]
 
text = 'A class is a blueprint for the object.'
 
print("1-gram: ", extract_ngrams(text, 1))
print("2-gram: ", extract_ngrams(text, 2))
print("3-gram: ", extract_ngrams(text, 3))
print("4-gram: ", extract_ngrams(text, 4))


# ## N-gram using TextBlob

# In[20]:


from textblob import TextBlob
 
# Function to generate n-grams from sentences.
def extract_ngrams(data, num):
    n_grams = TextBlob(data).ngrams(num)
    return [ ' '.join(grams) for grams in n_grams]
 
text = 'A class is a blueprint for the object.'
 
print("1-gram: ", extract_ngrams(text, 1))
print("2-gram: ", extract_ngrams(text, 2))
print("3-gram: ", extract_ngrams(text, 3))
print("4-gram: ", extract_ngrams(text, 4))


# In[ ]:




