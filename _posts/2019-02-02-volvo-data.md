---
title: "Volvo sales data - Explanatory data analysis and Time series modelling"
date: 2019-02-02
tags: [time series, LSTM, explanatory analysis]
header:
  image: "/volvo_data/207937_Volvo_Cars_T8_Twin_Engine_Range.jpg"
excerpt: "Time Series, LSTM Network, Explanatory Analysis"
mathjax: "true"
---

## Background

Vovlo Car Group finalized the sales results of 2018 year in [official release](https://www.media.volvocars.com/global/en-gb/media/pressreleases/247393/volvo-cars-sets-new-global-sales-record-in-2018-breaks-600000-sales-milestone) and hit the global record of ***600.000*** sales.<br>

The major contribution to overall sales growth was driven by China **(14.1%)** and US **(20.6%)**.<br>
In comparison with December'17 we see the slight demand slowdown in Europe region **(-1.3%)** and US **(-8.8%)**. However, total December volumes represent sustainable growth year over year. <br>

To effectively predict auto sales and improve Volvo Group competitiveness we will analyze the monthly data and, 
revealing seasonal flactuations derive predictions powered by neural network.

## Main goal

Analyze monthly sales data and build predictive model according to listed steps:
 - Explore sales data and establish ETL into Google BigQuery;
 - Perform Explanatory data analysis of time series;
 - Model preparation and evaluating;
 - Provide recommendataions for further improvement. 
 
 Entire process can be illustrated by diagram:
 
 ![LSTM]({{ 'volvo_data/workflow.png' | absolute_url }})


## Data

Data we will use for this project is published on Volvo Car Group [official website](https://www.media.volvocars.com/global/en-gb/corporate/sales-volumes) and contains:
- monthly **sales volumes**;
- auto **model**;
- respected time period - **month, year**.

To be consistant we will add respected time period column in following format: **"YYYY-MM-DD"**.

## Data preprocessing 

### Step 1: Primary data cleaning

Initial CSV dataset is availbale in repository via the [link](https://github.com/MakarovArtyom/side_projects/blob/master/sales_full.csv)
We start with data transformation into suitable for analysis format: 
- Convert sales amount column in "int" format;
- Retrieve month and year stamps from date columns;
- Write resulted columns into .csv file. 

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- import librariies for data manipulation  
- split date into month/year
- write dataframe into new csv 

"""

import pandas as pd
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

# reading csv file
df= pd.read_csv('sales.csv', sep=';', encoding='utf-8')
df.drop('Unnamed: 5', axis=1, inplace=True)
# convert to str and numeric
df['sales'] = df['sales'].astype('str')
df['sales']=pd.to_numeric(df['sales'])
# date convertion
df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
df['date_year'] = [x.strftime('%Y')for x in df['date']]
df['date_month'] = [x.strftime('%m')for x in df['date']]
# reading into csv
df=df[['model', 'date', 'sales', 'date_year', 'date_month']]
df.to_csv('sales_full.csv', encoding='utf-8')
 ```
 
 </p>
</details>

### Step 2: Store data with Google Cloud Storage

On this stage we upload data into Cloud Storage to establish data transfer over all Google services. 
Since the files are uploaded on GitHub we can use Git to clone repository and store entire folder. 
Using Google Shell type following command:

 ```python
 $ git clone https://github.com/MakarovArtyom/side_projects.git
 $ cd side_projects/volvo_data
 $ head sales_all.csv
  ```
Explore the .csv in Output:

 ```python
,model,date,sales,date_year,date_month0,S60 CC,2018-12-31,87,2018,121,S60 II,2018-12-31,961,2018,122,S60L,2018-12-31,1314,2018,123,S80 II,2018-12-31,0,2018,124,S90,2018-12-31,5108,2018,125,V40,2018-12-31,5121,2018,126,V40 CC,2018-12-31,1591,2018,127,V60,2018-12-31,524,2018,128,V60 CC,2018-12-31,249,2018,12
 ```
Next we need to create a "bucket" and store the files (additionally we create folder 'sql'):
```python
$ gsutil cp cloudsql/* gs://<BUCKET-NAME>/sql/ 
```

### Step 3: Upload into Google BigQuery 

On this stage we will load .csv file into BigQuery, create a table and query data. 
Start with dataset creation:
![LSTM]({{ 'volvo_data/create_dataset.PNG' | absolute_url }})

Continue with creating a table, we load sales_all.csv from storage, specifiyng the name - "sales_monthly".
![LSTM]({{ 'volvo_data/create_table.PNG' | absolute_url }})

Finally, use SQL syntax to query a table:
 ```python
 SELECT model, date, sales, date_year, date_month
FROM
  [prredictions:sales_volvo.sales_monthly]
LIMIT
  1600
  ```
![LSTM]({{ 'volvo_data/query.PNG' | absolute_url }})

## Loading dataset to Notebook

Now we are ready to import and analyze time series in Jupyter notebook. 
We willplotly to visualize the data, pandas package for data manupulation and keras for model building.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- import Plotly modules for interactive data visualization 
- connect to plot inside notebook
"""

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
import colorlover as cl
import cufflinks as cf
init_notebook_mode(connected=True)

# data manipulation libraries
import pandas as pd  
from pandas.io import gbq # retrieve google big query data
import numpy as np           
import matplotlib.pyplot as plt 
%pylab inline
plt.style.use('seaborn-whitegrid')

# will ignore the warnings
import warnings                 
warnings.filterwarnings("ignore")

# keras to build LSTM network 
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
 ```
 
 </p>
</details>

Once we're done we can start to load and explore dataset.
 ```python
 sql = """
    SELECT *
    FROM `sales_volvo.sales_monthly`
    LIMIT 1600
 # Run a Standard SQL query with the project set explicitly
project_id = 'prredictions'
sales = pd.read_gbq(sql, project_id=project_id, dialect='standard')
sales.head()
"""
  ```
![LSTM]({{ 'volvo_data/dataset.PNG	' | absolute_url }})

## Explonatory analysis

Starting with listing down a number of variables affecting amount of sales:

1. Sales **per month** - the sales amount can differ depends on month;
2. Sales **per year** - the sales amount can flactuate depending on year;
3. Sales **per model** - particular Volvo model represents differnet sales amount in time.  

### Time period effect

On the graph we see the tendency of average sales amount **reaches its peak** at the **last month of each quarter**. 

<iframe width="800" height="600" frameborder="0" scrolling="no" src="//plot.ly/~makarovartyom/3.embed"></iframe>

Besides, we see the positive trend per year with the **highest total sales results in 2018**.

<iframe width="800" height="600" frameborder="0" scrolling="no" src="//plot.ly/~makarovartyom/5.embed"></iframe>

### Volvo sales per model

Then we can estimate the mean sales per model and get the top performers over the years.
<iframe width="800" height="600" frameborder="0" scrolling="no" src="//plot.ly/~makarovartyom/9.embed"></iframe>




