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
 
