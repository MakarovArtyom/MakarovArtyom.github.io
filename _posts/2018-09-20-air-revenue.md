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
 ```
 
 </p>
</details>
