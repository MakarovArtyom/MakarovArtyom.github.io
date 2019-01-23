---
title: "Google BigQuery Taxi Fare Predictions"
date: 2018-08-28
tags: [data science, regression, xgboost]
header:
  image: "/images/taxi_pic.jpg"
excerpt: "Data Science, Regression, XGBoost"
mathjax: "true"
---

# Introduction

For this project we are going to use Google BigQuery data to predict the estimated fare amount of New York taxi rides. 
We aim to manipulate the dataset, prepare exploratory analysis, retreiving all the hidden patterns and variables relationships for creating machine learning models to offer expected fare.
The major part is consentrated on data cleaning, visual component and fetaure engineering.

#### The Data:

The data was collcted via Google BigQuery from "NYC TLC Trips" public dataset and contains information about NYC taxi trips details:

- Pickup longitude/latitude;
- Dropoff longitude/latitude;
- Pickup/dropoff time;
- Passenger count;
- Trip distance;
- Fare amount. 

We will retrieve 2015 year data and load 2 millions rows into dataframe. 

### Libraries import

First we need to import all libraries we will use to manipulate dataframe, visualize results and make predictions. 

<details><summary>Python code</summary> 
  
<p>
  
 ```python
 """
- gbq - load BigQuery data  
- pandas, matplotlib, numpy - for data manipulation 
- seaborn - data visualization

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.io import gbq
import boto3 # transfer file with data to S3 cloud
%matplotlib inline
    
"""
- dateteme module for time type convertation  
- we will use math modules to derive the custom distance feature based on radians

"""
import datetime as dt
from math import sin, cos, sqrt, atan2, radians
from scipy import stats
from sklearn.utils import shuffle

"""
- import Plotly modules for interactive data visualization 
- connect to plot inside notebook

"""

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go

init_notebook_mode(connected=True)

"""
- sklearn for model building and performance validation

"""

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
```
</p>
</details>

### Data processing

We start with collection data from Google BigQuery public dataset and writing this into .csv file. 
Optionally we will store the file on AWS cloud. 

<details><summary>Python code</summary> 
  
<p>
  
```python
"""
- loading data from big query project, write into frame 
- automatically upload csv file to S3 storage 

"""

df = gbq.read_gbq('SELECT * FROM taxi.taxi_fare_2015 LIMIT 2000000', project_id='XXXXXXX')
df.to_csv('fares_all.csv')

ACCESS_ID = 'XXXXXX'
ACCESS_KEY='XXXXXX'
filename = 'fares_all.csv'
bucket_name = 'storagebucketmachinelearning'


s3 = boto3.client('s3', aws_access_key_id=ACCESS_ID,
         aws_secret_access_key= ACCESS_KEY)
s3.upload_file(filename, bucket_name, filename)

# select the columns we need
cols=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
  'passenger_count', 'pickup_datetime', 'dropoff_datetime', 'fare_amount']
df=df[cols]
```

</p>
</details>

Resulted dataframe is presented below:
<img src="taxi_output/data_frame.PNG" alt="hi" class="inline"/>









## H2 Heading

### H3 Heading

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$
