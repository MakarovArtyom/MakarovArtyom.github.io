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

The major contribution to overall sales growth was driven by China ***(14.1%)*** and US (***20.6%***). In comparison with December'17 we see the slight demand slowdown in Europe region (***-1.3%***) and US (***-8.8%***). However, total December volumes represent sustainable growth year over year. <br>

To effectively predict auto sales and improve Volvo Group competitiveness we will analyze the monthly data and, 
revealing seasonal flactuations derive predictions powered by neural network.

## Main goal:

Analyze monthly sales data and build predictive model according to listed steps:
 - Explore sales data and establish ETL into Google BigQuery;
 - Perform Explanatory data analysis of time series;
 - Model preparation and evaluating;
 - Provide recommendataions for further improvement. 
 
 Entire process can be illustrated by diagram:
 
 ![LSTM]({{ 'volvo_data/workflow.png' | absolute_url }})
