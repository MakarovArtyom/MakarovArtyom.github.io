---
title: "Time Series for Air revenue passenger prediction"
date: 2018-10-11
tags: [time series, statistical models, revenue prediction]
header:
  image: "/air_rev_output/air_revenue.jpeg"
excerpt: "Time series, Statistical models, Revenue prediction"
mathjax: "true"
---

## Overview

Under this project scope we will forecast monthly air passenger revenue using time series techniques. <br>
Time series is useful method in machine learning when we need to forecast a value given a time component: an information about the time period previous values were recorded. 

**Data**: series sourced form "Federal reserve economic data" via Quandl API:<br>
https://www.quandl.com/data/FRED/AIRRPMTSI-Air-Revenue-Passenger-Miles. <br>
Observations include *monthly* air revenue starting from 2000 to present. 

### Import libraries

For this project we will need Quandl limbrary to retreive the data and statsmodels,api to work with statistical models.
Throughout the analysis, we will visualize results with matplotlib and plotly libraries.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
import quandl # use quandl linbrary to retreive the data 

"""
- will use pandas and numpy for data manipulation and calculus
- import datetime to work with datetime type
- use Series library to work with time series 

"""
import pandas as pd          
import numpy as np           
from datetime import datetime    
from pandas import Series  

import matplotlib.pyplot as plt 
%pylab inline
plt.style.use('seaborn-whitegrid')

# will ignore the warnings
import warnings                 
warnings.filterwarnings("ignore")

"""
- to work with statistical models we statsmodels.api and stats module from scipy
- use itertools and seasonal_decompose for data preparation and model building
- to plot PACF/ACF chart use tsaplots graphic module

"""
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import Holt

import itertools
from itertools import product
from statsmodels.tsa.seasonal import seasonal_decompose # perform seasonal decomposition

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

"""
- visualize results using plotly library, connected to current notebook

"""
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True)
 ```
 
 </p>
</details>


### Data processing and EDA

To retreive data from source, we opearte .get() function, using token and dataset reference.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
authtoken = "XXXXXX"
df = quandl.get("FRED/AIRRPMTSI", authtoken=authtoken)
df.head() # show how the data looks like 

 ```
 
 </p>
</details>

![LSTM]({{ 'air_rev_output/series.PNG' | absolute_url }})

In practice it's not necessary to use the entire time series. To provide precise analysis we are going to use date range from 2010-01-01 to 2018-09-01.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
df=df['2010-01-01':'2018-09-01']

# plot values from range
fig, ax = plt.subplots(figsize=(15,7)) # setting up size
df.Value.plot() # graph plot
plt.ylabel('Value')
plt.title('Monthly air revenue')
plt.show()

 ```
 
 </p>
</details>

![LSTM]({{ 'air_rev_output/monthly_air.PNG' | absolute_url }})

**Stationarity test**

We see our series are characterized by systematically repeating cycles. To gain high performance of predictions we need to make them stationary to build and apply models afterwards. 
Then, we start with series decomposition using seasonal component and Dickey-Fuller stationarity test.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- plot decomposed series, display Dickey-Fuller criteria applying  .adfuller() function 

"""

plt.figure(figsize(15,10))
sm.tsa.seasonal_decompose(df.Value).plot()
print "Dickey-Fuller criteria: p=%f" % sm.tsa.stattools.adfuller(df.Value)[1]
 ```
 
 </p>
</details>

 ```python
Dickey-Fuller criteria: p=0.999088
 ```
![LSTM]({{ 'air_rev_output/stationarity_1.PNG' | absolute_url }})
![LSTM]({{ 'air_rev_output/stationarity_2.PNG' | absolute_url }})

The high coefficient value for Dickey-Fuller criteria proves non-stationarity on 5% significance level.<br>
The residuals and seasonality for decomposed series have a systematic character.

**Dispersion stabilization**

Additionally we are able to apply *Box-Cox power transformation* in order to reach higher dispersion stability. 

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- create additional column of tranformed values using Box-Cox method
- apply Dickey-Fuller test for transformed values 

"""
df['Value_t'], lam=stats.boxcox(df['Value'])
plt.figure(figsize(13,5))
df['Value_t'].plot()
plt.ylabel(u'FFE_amount-transformed')
print "lambda-optimum: %f" % lam
print "Dickey-Fuller criteria: p=%f" % sm.tsa.stattools.adfuller(df['Value_t'])[1]
 ```
 
 </p>
</details>

 ```python
lambda-optimum: -0.039101
Dickey-Fuller criteria: p=0.999030
  ```
    
![LSTM]({{ 'air_rev_output/dispersion_stab.PNG' | absolute_url }})


**Differencing**

Whereas the With Box-Cox transformation applied supposed to result stationary process based on Dickey-Fuller criteria, we still can observe non-ramdomized cycles in time series.<br>
To avoid systematic flactuations, we can try differencing operation.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- applying 12 order differencing, we reduce data 
- however, this operation will improve the stationarity ratio 

"""

df['diff'] = df.Value_t.diff(12)
plt.figure(figsize(15,10))
sm.tsa.seasonal_decompose(df['diff'][12:]).plot()
print "Dickey-Fuller criteria: p=%f" % sm.tsa.stattools.adfuller(df['diff'][12:])[1]
 ```
 
 </p>
</details>

 ```python
 Dickey-Fuller criteria: p=0.599792
  ```
![LSTM]({{ 'air_rev_output/differencing.PNG' | absolute_url }})
![LSTM]({{ 'air_rev_output/differencing_2.PNG' | absolute_url }})

Based on Dickey-Fuller criteria we can't prove stationarity, but we are able to make differencing for new series.

