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

```python
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from pandas.io import gbq
    import boto3 # transfer file with data to S3 cloud 

    %matplotlib inline
```



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
