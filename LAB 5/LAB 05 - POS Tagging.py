#!/usr/bin/env python
# coding: utf-8

# ## POS-Tagger

# ### NLTK Tagger

# #### A part-of-speech tagger, or POS-tagger, processes a sequence of words, and attaches a part of speech tag to each word.

# In[1]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

text1 = word_tokenize("And now for something completely different")
print(nltk.pos_tag(text1))
print()

text2 = word_tokenize("They refuse to permit us to obtain the refuse permit")
print(nltk.pos_tag(text2))
print()

#to get the meaning of the tags
nltk.help.upenn_tagset('JJ')

text3 = word_tokenize("The back door")
print(nltk.pos_tag(text3))
print()

text4 = word_tokenize("I couldn’t get back to sleep")
print(nltk.pos_tag(text4))
print()


# #### The text.similar() method takes a word w, finds all contexts w1 w w2, then finds all words w' that appear in the same context, i.e. w1 w' w2.

# In[2]:


text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
word_list = ['woman', 'bought', 'over', 'the']
for w in word_list:
    print("\nwords in text similar to '"+ w + "' are: ")
    text.similar(w)


# ### Representing Tagged Tokens

# #### By convention in NLTK, a tagged token is represented using a tuple consisting of the token and the tag. We can create one of these special tuples from the standard string representation of a tagged token, using the function str2tuple():

# In[3]:


tagged_token = nltk.tag.str2tuple('fly/NN')
print(tagged_token)
print(tagged_token[0])
print(tagged_token[1])

sent = '''
    The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
    other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
    Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
    said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
    accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
    interest/NN of/IN both/ABX governments/NNS ''/'' ./.
    '''
print(sent)


# ## Reading Tagged Corpora

# #### Several of the corpora included with NLTK have been tagged for their part-of-speech.

# In[4]:


nltk.corpus.brown.tagged_words()
nltk.corpus.brown.tagged_words(tagset='universal')
# Here are some more examples, again using the output format illustrated for the Brown Corpus:
print(nltk.corpus.nps_chat.tagged_words())
print()
print(nltk.corpus.conll2000.tagged_words())
print()
print(nltk.corpus.treebank.tagged_words())
print()

# Not all corpora employ the same set of tags. Initially we want to avoid the complications of these tagsets, 
# so we use a built-in mapping to the "Universal Tagset"
print(nltk.corpus.brown.tagged_words(tagset='universal'))
print(nltk.corpus.treebank.tagged_words(tagset='universal'))
print()

'''Tagged corpora for several other languages are distributed with NLTK, including Chinese, Hindi, Portuguese, Spanish, 
Dutch and Catalan. These usually contain non-ASCII text, and Python always displays this in hexadecimal when printing a 
larger structure such as a list.'''
print(nltk.corpus.sinica_treebank.tagged_words())
print(nltk.corpus.indian.tagged_words())
print(nltk.corpus.mac_morpho.tagged_words())
print(nltk.corpus.conll2002.tagged_words())
print(nltk.corpus.cess_cat.tagged_words())


# ## Universal Part-of-Speech Tagset

# #### Let's see which of these tags are the most common in the news category of the Brown corpus:

# In[5]:


from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.most_common()


# ## Nouns

# #### Nouns generally refer to people, places, things, or concepts, e.g.: woman, Scotland, book, intelligence. Nouns can appear after determiners and adjectives, and can be the subject or object of the verb

# In[6]:


word_tag_pairs = nltk.bigrams(brown_news_tagged)
noun_preceders = [a[1] for (a, b) in word_tag_pairs if b[1] == 'NOUN']
fdist = nltk.FreqDist(noun_preceders)
[tag for (tag, _) in fdist.most_common()]


# ## Verbs

# #### Verbs are words that describe events and actions, e.g. fall, eat. In the context of a sentence, verbs typically express a relation involving the referents of one or more noun phrases.

# In[7]:


wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd = nltk.FreqDist(wsj)
[wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'VERB']


# #### Note that the items being counted in the frequency distribution are word-tag pairs. Since words and tags are paired, we can treat the word as a condition and the tag as an event, and initialize a conditional frequency distribution with a list of condition-event pairs. This lets us see a frequency-ordered list of tags given a word:

# In[8]:


cfd1 = nltk.ConditionalFreqDist(wsj)
cfd1['yield'].most_common()
cfd1['cut'].most_common()


# #### We can reverse the order of the pairs, so that the tags are the conditions, and the words are the events. Now we can see likely words for a given tag. We will do this for the WSJ tagset rather than the universal tagset:

# In[9]:


wsj = nltk.corpus.treebank.tagged_words()
cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
list(cfd2['VBN'])


# #### To clarify the distinction between VBD (past tense) and VBN (past participle), let's find words which can be both VBD and VBN, and see some surrounding text:

# In[10]:


[w for w in cfd1.conditions() if 'VBD' in cfd1[w] and 'VBN' in cfd1[w]]
idx1 = wsj.index(('kicked', 'VBD'))
wsj[idx1-4:idx1+1]
idx2 = wsj.index(('kicked', 'VBN'))
wsj[idx2-4:idx2+1]


# ## Exploring Tagged Corpora

# #### Exploring Tagged Corpora

# In[11]:


brown_learned_text = brown.words(categories='learned')
sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == 'often'))


# #### However, it's probably more instructive use the tagged_words() method to look at the part-of-speech tag of the following words:

# In[12]:


brown_lrnd_tagged = brown.tagged_words(categories='learned', tagset='universal')
tags = [b[1] for (a, b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'often']
fd = nltk.FreqDist(tags)
fd.tabulate()


# #### Notice that the most high-frequency parts of speech following often are verbs. Nouns never appear in this position (in this particular corpus). 

# #### Next, let's look at some larger context, and find words involving particular sequences of tags and words (in this case "Verb to Verb"). In code-three-word-phrase we consider each three-word window in the sentence, and check if they meet our criterion. If the tags match, we print the corresponding words.

# In[13]:


import nltk
from nltk.corpus import brown
def process(sentence):
    for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence): 
        if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')): 
            print(w1, w2, w3)
            
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
data = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in brown_news_tagged)

for word in sorted(data.conditions()):
    if len(data[word]) > 3:
        tags = [tag for (tag, _) in data[word].most_common()]
        print(word, ' '.join(tags))


# ## Automatic Tagging

# In[14]:


from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
print(brown_tagged_sents)
print()
print(brown_sents)
print()


# ## The Default Tagger

# #### The simplest possible tagger assigns the same tag to each token. This may seem to be a rather banal step, but it establishes an important baseline for tagger performance. In order to get the best result, we tag each word with the most likely tag. Let's find out which tag is most likely (now using the unsimplified tagset):

# In[15]:


tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
dt = nltk.FreqDist(tags).max()
print(dt)
print("~~~~~~~~~ Example ~~~~~~~~~")
raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)


# #### Unsurprisingly, this method performs rather poorly. On a typical corpus, it will tag only about an eighth of the tokens correctly, as we see below:

# In[16]:


default_tagger.evaluate(brown_tagged_sents)


# ## The Lookup Tagger

# #### A lot of high-frequency words do not have the NN tag. Let's find the hundred most frequent words and store their most likely tag. We can then use this information as the model for a "lookup tagger" (an NLTK UnigramTagger):

# In[17]:


import nltk
from nltk.corpus import brown
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
baseline_tagger1 = nltk.UnigramTagger(model=likely_tags)
baseline_tagger1.evaluate(brown_tagged_sents)


# In[18]:


'''It should come as no surprise by now that simply knowing the tags for the 100 most frequent words enables us to tag a 
large fraction of tokens correctly (nearly half in fact). Let's see what it does on some untagged input text:'''

sent = brown.sents(categories='news')[3]
baseline_tagger1.tag(sent)


# In[19]:


'''Many words have been assigned a tag of None, because they were not among the 100 most frequent words. In these cases we 
would like to assign the default tag of NN. In other words, we want to use the lookup table first, and if it is unable to 
assign a tag, then use the default tagger, a process known as backoff. We do this by specifying one tagger as a parameter to 
the other, as shown below. Now the lookup tagger will only store word-tag pairs for words other than nouns, and whenever it 
cannot assign a tag to a word it will invoke the default tagger.'''

baseline_tagger2 = nltk.UnigramTagger(model=likely_tags,backoff=nltk.DefaultTagger('NN'))
baseline_tagger2.tag(sent)


# #### Put all this together:

# In[20]:


import nltk
def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))
def display():
    import pylab # install matplotlib
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()
display()  


# ## The Regular Expression Tagger

# #### The regular expression tagger assigns tags to tokens on the basis of matching patterns. For instance, we might guess that any word ending in ed is the past participle of a verb, and any word ending with 's is a possessive noun. We can express these as a list of regular expressions:

# In[21]:


patterns = [
     (r'.*ing$', 'VBG'),               # gerunds
     (r'.*ed$', 'VBD'),                # simple past
     (r'.*es$', 'VBZ'),                # 3rd singular present
     (r'.*ould$', 'MD'),               # modals
     (r'.*\'s$', 'NN$'),               # possessive nouns
     (r'.*s$', 'NNS'),                 # plural nouns
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN'),                    # nouns (default)
     (r'^\d+$', 'CD'),
     (r'.*ing$', 'VBG'),               # gerunds, i.e. wondering
     (r'.*ment$', 'NN'),               # i.e. wonderment
     (r'.*ful$', 'JJ')                 # i.e. wonderful
 ]

regexp_tagger = nltk.RegexpTagger(patterns)
tagger=nltk.tag.sequential.RegexpTagger(patterns)

import nltk
from nltk.tokenize import word_tokenize
text1 = word_tokenize('Python is a high-level, general-purpose programming language')
print(tagger.tag(text1))
print()
print(nltk.pos_tag(text1))


# ## Text Blob Tagger

# In[22]:


import nltk
from textblob import TextBlob
wiki = TextBlob("Python is a high-level, general-purpose programming language. Python is a high-level, general-purpose programming language.")
print(wiki.tags)

import nltk
from nltk.tokenize import word_tokenize
print()
text1 = word_tokenize("Python is a high-level, general-purpose programming language. Python is a high-level, general-purpose programming language.")
print(nltk.pos_tag(text1))


# In[23]:


import nltk
from textblob import TextBlob
wiki = TextBlob("Programming skill is very important for a analytics person.")
print(wiki.tags)
print(wiki.noun_phrases)
print()

import nltk
from nltk.tokenize import word_tokenize
text1 = word_tokenize("Programming skill is very important for a analytics person.")
print(nltk.pos_tag(text1))
print()

text2 = word_tokenize("what men?")
print(nltk.pos_tag(text2))
print()
print(nltk.help.upenn_tagset("VBZ"))
print(nltk.help.upenn_tagset("WP"))


# #### Lemmatization with POS Tags Specifications

# In[24]:


import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# Init Lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize Single Word with the appropriate POS tag
word = 'feet'
print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

# Lemmatize a Sentence with the appropriate POS tag
sentence = "The striped bats are hanging on their feet for best"
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])


# In[ ]:




