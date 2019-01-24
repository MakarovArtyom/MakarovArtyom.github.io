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

### The Data:

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
#### Step №1 - Data retrieving 

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

![LSTM]({{ 'taxi_output/data_frame.PNG' | absolute_url }})

#### Step №2 - Primary data cleaning

Next we will drop NaN values from dataframe and make sure that our target variable, "fare amount", takes postive values. 
Note, that some varibales, such as pickup or dropoff datetime should be converted into timestamp.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- drop rows with negative and zero fare amount
- drop NaN values 

"""
df=df[df['fare_amount']>0]
df=df.dropna(how='any')
df.isna().any()

"""
- apply the date type transformation for pickups/dropoffs

"""
df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df['dropoff_datetime'] = df['dropoff_datetime'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

 ```
 </p>
</details>

Now let's look at results of timestamp transformation: 

![LSTM]({{ 'taxi_output/dtypes.PNG' | absolute_url }})

#### Step №3 - Features engineering

The further steps will be to calculate the difference in seconds and create new column for taxi trips duration - *"diff"*. <br> 
Besides we are going to derive time periods from stamps for both dropoffs and pickups. These variables will be used for further hypothesis testing and models fitting.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
 """
- convert the diff columns into 'float' type

"""
df['diff'] = df['dropoff_datetime'] - df['pickup_datetime']
df['diff'] = df['diff'].astype('timedelta64[s]')
 
 """
- create a separate column for each period of time, when dropoff/pickup was occured

"""
df['dropoff_year'] = [x.strftime('%Y')for x in df['dropoff_datetime']]
df['dropoff_month'] = [x.strftime('%m')for x in df['dropoff_datetime']]
df['dropoff_day'] = [x.strftime('%d')for x in df['dropoff_datetime']]
df['dropoff_hour'] = [x.strftime('%H')for x in df['dropoff_datetime']]


df['pickup_year'] = [x.strftime('%Y')for x in df['pickup_datetime']]
df['pickup_month'] = [x.strftime('%m')for x in df['pickup_datetime']]
df['pickup_day'] = [x.strftime('%d')for x in df['pickup_datetime']]
df['pickup_hour'] = [x.strftime('%H')for x in df['pickup_datetime']]
 ```
 </p>
</details>

Display how the new features look like: 

![LSTM]({{ 'taxi_output/new_features.PNG' | absolute_url }})

The distance between pickup and dropoff points can also be highly useful for our analysis. To estimate the distance we will apply *Haversine formula* based on radians and add related column to a dataframe as *"distance_trip"* variable.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- difine an approxiamte earth radius in km
- write a function to convert longitude/latitude to radians and return km distance from formula

"""

R=6373.0
def distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # estimate the distance between dropoff (lon2, lat2) and pickup (lon1, lat1)
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
    
# adding a column to dataframe 
df['distance_trip'] = distance(df['pickup_longitude'], df['pickup_latitude'], 
                               df['dropoff_longitude'], df['dropoff_latitude'])
 ```
 </p>
</details>

Note, that calculated distance is presented in kilometers and has a float type: 

![LSTM]({{ 'taxi_output/distance.PNG' | absolute_url }})

#### Step №4 - Outliers detection 

On the next step we continue data cleaning with outliers check. Below we investigate the risk of incorrect spatial data - all coordinates pairs should be limited by boundaries: *[-90, 90]* for latitude and *[-180, 180]* for longitude.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- initialize the interval for latitude/longitude
- check if dataframe contains values out of boundaries, display the length

"""

lat_range=[-90,90]
long_range=[-180,180]

# pickups check
print "dataframe length: "+ str(len(df[(df['pickup_latitude']> lat_range[0]) | (df['pickup_latitude']< lat_range[1])]))
print "dataframe length: "+ str(len(df[(df['pickup_longitude']> long_range[0]) | (df['pickup_longitude']< long_range[1])]))
# dropoffs check
print "dataframe length: "+ str(len(df[(df['dropoff_latitude']> lat_range[0]) | (df['dropoff_latitude']< lat_range[1])]))
print "dataframe length: "+ str(len(df[(df['dropoff_longitude']> long_range[0]) | (df['dropoff_longitude']< long_range[1])]))
 ```
  </p>
</details>



All pairs of coordinates lie inside of corresponding intervals. 

![LSTM]({{ 'taxi_output/frame_length.PNG' | absolute_url }})

Apart from data correctness check the useful option will be to remove extreme points for distance, duration and fare amount variables.
This operation will reduce the variance and take out potential outliers from data.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- we will use numpy .percentile function to calculate first and third quartile 
- next calculate interquartile interval 
- set up the boundaries for extremely small and large values 
- return the frame inside interval 

"""
def remove_outliers(df, column):
    quartile_1, quartile_3 = np.percentile(df[column], [25, 75])
    iqr = quartile_3 - quartile_1
    min = quartile_1 - (iqr * 1.5)
    max = quartile_3 + (iqr * 1.5)

    df = df[(df[column] <= max) & (df[column] >= min)]
    return df
    
 """
- apply function for duration, distance and fare amount 

"""
data=remove_outliers(df, 'distance_trip')
data=remove_outliers(data, 'fare_amount')
data=remove_outliers(data, 'diff')    
 ```
  </p>
</details>





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
