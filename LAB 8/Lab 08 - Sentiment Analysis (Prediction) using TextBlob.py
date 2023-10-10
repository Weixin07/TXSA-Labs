# Sentiment Prediction using TextBlob
from textblob import TextBlob
reviews = input("Enter the feedback: ")
sent = TextBlob(reviews)
print(sent.polarity)


# Consider every sentence in the text input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from textblob import TextBlob
reviews = input("Enter the feedback: ")
sent = TextBlob(reviews)
for s in sent.sentences:
    print(s)
    print("The polarity is ", s.polarity)
    if (s.polarity == 0):
        print ("The sentiment is Neutral")
    else:
        if (s.polarity > -0):
            print ("The sentiment is Positive")
        else:
            print("The sentiment is Negative")


# Consider it as overall text input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from textblob import TextBlob
reviews = input("Enter the feedback: ")
sent = TextBlob(reviews)
print("The polarity is ", sent.polarity)
if (sent.polarity == 0):
    print ("The sentiment is Neutral")
else:
    if (sent.polarity > -0):
        print ("The sentiment is Positive")
    else:
        print("The sentiment is Negative")


# Perform sentiment prediction for the reviews stored in a CSV file ~~~~~~~~~~~ 
from textblob import TextBlob
import pandas as pd 
data = pd.read_csv("D:/APU/TXSA-CT107-3-3/LAB/LAB 8/opinion_dataset.csv") 
data.head()
opinions = data['Opinions']
opinions.count()

for line in opinions:
    print(line)
    sent = TextBlob(line)
    print("The polarity is ", sent.polarity)
    if (sent.polarity == 0):
        print ("The sentiment is Neutral")
    else:
        if (sent.polarity > -0):
            print ("The sentiment is Positive")
        else:
            print("The sentiment is Negative")
    print()



# Converting a CSV file into a text file ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import csv
csv_file = input('Enter the name of your input file: ')
txt_file = input('Enter the name of your output file: ')
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()
    




