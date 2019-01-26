---
title: "Sentiment analysis - London restaurants reviews"
date: 2018-12-05
tags: [sentiment analysis, logistic regression, adaboost]
header:
  image: "/yelp_output/london_pic.jpeg"
excerpt: "Sentiment analysis, Logistic regression, Adaptive Boosting"
mathjax: "true"
---

# Overview

The goal of this project is to analyze the customers' sentiments based on Yelp restaurant reviews.<br>
This type of analysis is known as a part of NLP tasks and highly used for business purposes, such as online marketing automation, measurement of product's users exeperience and etc.<br>
We will use London restaurant reviews, scrapped from [Yelp](https://www.yelp.com/) website.
In this particular task we will follow the steps below to complete end-to-end machine learning project:<br>

1. Import necessary libraries;
2. Data gathering and cleansing;
3. Explanatory analysis;
4. Models building and result comparison;
5. Making prediction.

The closing section sums up the project results and aprovides the further improvement recommendations. 

## Libraries import

First we will use *Beautifulsoup* library to scrap data. Next we store reviews with corresponding ratings to pandas frame and visualize with wordcloud amd matplotlib. 
Finally we import scikit-learn modules for models building and evaluation. 

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- importing urllib2, BeautifulSoup and requests for yelp web-scrapping 
- pandas, matplotlib, numpy - for data manipulation 

"""
import urllib2 
from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

"""
- libraries for working with text and visualizing output

"""
from collections import Counter
import nltk
from wordcloud import WordCloud
from nltk import FreqDist # get word count
from nltk.tokenize import word_tokenize # tokenize the sentences by word

"""
- model building and measure performance
- parametres are estimated with grid search 

"""
from sklearn import metrics
from sklearn.utils import shuffle # shuffle data before learning curve plotting
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import scikitplot as skplt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

 ```
 
 </p>
</details>

## Data preparation 

The restaurants listed on Yelp wibsite and used for reviews collecting.

 ```python
 restaurants = [
'https://www.yelp.com/biz/hide-london?start=', 
'https://www.yelp.com/biz/a-wong-london?start=', 
'https://www.yelp.com/biz/restaurant-gordon-ramsay-london-3?start=', 
'https://www.yelp.com/biz/dinner-by-heston-blumenthal-london?start=', 
'https://www.yelp.com/biz/the-breakfast-club-london-3?start=', 
'https://www.yelp.com/biz/five-guys-london-23?start=', 
'https://www.yelp.com/biz/kowloon-london?osq=Kowloon+Restaurant?start=', 
'https://www.yelp.com/biz/cubana-london?start=', 
'https://www.yelp.com/biz/pizza-east-london?start=', 
'https://www.yelp.com/biz/las-iguanas-london?start=', 
'https://www.yelp.com/biz/mother-mash-london?start=', 
'https://www.yelp.com/biz/shake-shack-covent-garden-london-4?start=', 
'https://www.yelp.com/biz/meat-liquor-london?start=', 
'https://www.yelp.com/biz/lupita-london-9?start=',
'https://www.yelp.com/biz/jamies-italian-covent-garden-london?start=']
  ```
Here we will create the empty lists for reviews and corresponding ratings. 
With each link from restaurants list opened we will iterate through the range of pages and scrap reviews data.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
review_list=[]
rating_list=[]
pages=range(0,260,20)

# create a list of all pages for each restaurat
all_pages=[] 

for url in restaurants:
    for page in pages:
        rest_page = url+str(page)  
        all_pages.append(rest_page) # append the next page to list
        
 """
- open each link with urllib2
- read the content with Beautiful soup 
- find corresponding review text and rating under 'div' calss on html page 

"""

for link in all_pages:
    content = urllib2.urlopen(link) 
    soup = BeautifulSoup(content, 'html.parser')
    
    for review in soup.findAll('div',{"class":"review-content"}):
        rating_list.append(review.div.div.div.get("title"))
        review_list.append(review.find('p').text)
 ```
 
 </p>
</details>

Next we will initialize dataframe and add columns with both text and rating we scrapped on previous step.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
reviews = pd.DataFrame()
reviews['rating'] = rating_list
reviews['rating'] = reviews['rating'].str[:3]
reviews['reviews'] = review_list

"""
- remove punctuation from sentences
- lead all words to a lower case 
- convert rating to float and write into csv  to save results

"""
reviews['reviews_punctuation_free'] = reviews['reviews'].str.replace('[^\w\s]',' ') # used space, since some words can be merged
reviews['reviews_punctuation_free'] = reviews['reviews_punctuation_free'].str.lower()

reviews['rating'] = reviews['rating'].astype('float64')
reviews.to_csv('sent_reviews.csv', sep='\t', encoding='utf-8')
 ```
 
 </p>
</details>

### Assign sentiment classes

For each review in frame we assign the sentiment class in order to split the reviews on positive or negative and create a target label.

- rating = 4.0 and 5.0 --> ***class = 1***
- rating < 2.0 --> ***class = 0***
- rating = 3.0 --> *do not include, neutral assessment*

<details><summary>Python code</summary> 
  
<p>
  
 ```python
reviews = reviews[reviews['rating'] != 3.0]
reviews['sentiment'] = reviews['rating'].apply(lambda rating : +1 if rating >= 4.0 else 0)

# display how dataframe looks like
reviews.tail(10)

 ```
  </p>
</details>

![LSTM]({{ 'yelp_output/frame.PNG' | absolute_url }})
 
## Explanatory analysis

### Imbalance classes

Starting explanatory analysis with investigating balance classes we can plot the number of reviews assigned to each class.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
plt.style.use('bmh') # set up matplotlib style 

fig, ax = plt.subplots(figsize=(7,5))
reviews['sentiment'].value_counts().plot.bar() 
ax.set_xlabel('sentiment')
ax.set_title('Sentiment classes')

 ```
  </p>
</details>

![LSTM]({{ 'yelp_output/sent_classes.png' | absolute_url }})

<details><summary>Python code</summary> 
  
<p>
  
 ```python
# display the number of each class in percentage
(reviews['sentiment'].value_counts(normalize=True)*100).round(2)

 ```
  </p>
</details>


 ```python
1    78.07
0    21.93
Name: sentiment, dtype: float64
 ```
 
Bar chart above highlight significant imbalance classes with positive major class. 
One of the propriate ways to mitigate imbalance is to make the equal proportion for both classes. However, the obvious disadvantage of this approach leads to entire sample reduction.
 
<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- split frame into parts based on positive/negarite sentiment
- undersample the major class to keep equal size for both
- create dataframe with balanced classes

"""
positive_frame = reviews[reviews['sentiment']==1]
negative_frame = reviews[reviews['sentiment']==0]

# estimate the size ratio
percentage = len(negative_frame)/float(len(positive_frame))
negative = negative_frame
positive = positive_frame.sample(frac=percentage) # use the same percentage
# append negative reviews sample to reduced positive frame 
reviews_data = negative.append(positive)

print "Positive class ratio:", len(positive) / float(len(reviews_data))
print "Negative class ratio:", len(negative) / float(len(reviews_data))
print "Entire frame length:", len(reviews_data)

 ```
  </p>
</details>

 ```python
Positive class ratio: 0.5
Negative class ratio: 0.5
Entire frame length: 894
 ```
 
### Words count and visualization

For each review perform transformation into string type and generate the cloud of words using Wordcloud() module.


<details><summary>Python code</summary> 
  
<p>
  
 ```python
text_pos=str() # initialize strig for positive  frame
for rev in positive_frame['reviews_punctuation_free']:
    rev=str(rev)
    text_pos = text_pos+rev
    
# create wordcloud object with text_pos as an argument 
wordcloud = WordCloud(background_color='white', width=1600, height=800).generate(text_pos)
# plot as a figure
plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

 ```
  </p>
</details>

![LSTM]({{ 'yelp_output/positive.png' | absolute_url }})

Next we complete the same procedure for negative reviews.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
text_neg=str() # initialize strig for negative  frame
for word in negative_frame['reviews_punctuation_free']:
    word=str(word)
    text_neg = text_neg+word

wordcloud = WordCloud(background_color='white', width=1600, height=800).generate(text_neg)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

 ```
  </p>
</details>

![LSTM]({{ 'yelp_output/negative.png' | absolute_url }})


As we can conclude from charts the most common words display neutral customers sentiments and gereral term (such as 'food', 'place', 'london' or 'order', 'pizza') rather than feelings or particular experience. 
Let's tokenize the words and sort them with respect of frequency.


<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- create a token object as a list of words, count the occurence respectively using FreqDist() module
- display the frequency for number of words (250 in example)

"""

nltk.download('punkt')

from nltk import FreqDist # used later to plot and get count
from nltk.tokenize import word_tokenize # tokenizes our sentence by word

token = word_tokenize(text_pos)

fdist = FreqDist(token)

fdist.most_common(250)

 ```
  </p>
</details>

Here are some words from tokenized list - note that the most frequent words have neutral sentiment. 

 ```python
 
 [('the', 12530),
 ('and', 7823),
 ('a', 6438),
 ('i', 5877),
 ('to', 4624),
 ('was', 4117),
 ('of', 4014),
 ('it', 3675),
 ('in', 2903),
 ('is', 2552),
 ('for', 2512),
 ('with', 2324),
 ('you', 2100),
 ('but', 2009),
 ('we', 1983),
 ...
 ('recommend', 201),
 ('bar', 201),
 ('cake', 201),
 ('want', 199),
 ('being', 198),
 ('way', 197),
 ('ice', 197),
 ('ll', 196),
 ('loved', 196),
 ('over', 196),
 ('sweet', 196),
 ('who', 194),
 ('excellent', 193),
 ('pork', 193),
 ('off', 192),
 ('eat', 191),
 ('know', 191),
 ('gravy', 190),
 ('every', 189)
  ```
  
  Then display frequency for negative cases:
  <details><summary>Python code</summary> 
  
<p>
  
 ```python
token = word_tokenize(text_neg)
fdist = FreqDist(token)
fdist.most_common(250)

 ```
  </p>
</details>

 ```python
 [('the', 3745),
 ('and', 2194),
 ('to', 2019),
 ('a', 1827),
 ('i', 1751),
 ('was', 1358),
 ('it', 1118),
 ('of', 1050),
 ('we', 892),
 ('in', 840),
 ('for', 793),
 ('that', 691),
 ...
 ('thing', 48),
 ('dirty', 48),
 ('bland', 48),
 ('couple', 48),
 ('okay', 47),
 ('decent', 47),
 ('star', 47),
 ('already', 46),
 ('put', 46),
 ('since', 46),
 ('doesn', 46),
 ('want', 45),
 ('little', 45)]
  ```
  
## Model selection

Since the majority of words for both positive and negative samples represent neutral sentiment, the reasonable solution seemed to prepare the bunch of sensitive words as features to build the separating hyperplane and hit the better model performance.


  
