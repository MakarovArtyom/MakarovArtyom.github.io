---
title: "Traditional and Bayesian approaches to A/B testing"
date: 2018-10-11
tags: [bayesian statistics, a/b test, hypothesis testing]
header:
  image: "/AB_test_output/ab.jpeg"
excerpt: "Bayesian Statistics, A/B test, Hypothesis testing"
mathjax: "true"
---


### Traditional A/B test - breif intoduction

Performing taditional A/B testing, we need to deal with a number of approximations, that stand behind frequentist statistics approach.<br>
Generally, this approach assumes the parameters of distribution are fixed and data generated randomly with respect of maximum likelihood function: <br>
$$\hat\theta = argmax_\theta(X |\theta)$$

We consider mean that stands for "bellshape" center and standard deviation (width) as parametres of normal distribution. <br>
The probability in terms of frequentist statistics measures as the long-term frequency of event occurrence. However we don't actually know the real mean value - we know it's fixed and can be estimated from population sample only. <br>
Common techniques, that frequentists use to make an assumption about estimate are:<br>

- $H_0$ hypothesis formulation;
- Data collection;
- Test statistics calculation with respect of p-values;
- Confidence interval.

Theoretically we need to collect an infinite number of samples from population and constract CI (for istance, 95% probability) for each one, but in practice we draw a single sample and expect 95% of the interval estimates contain population parameter.

#### t-test: theoretical aspects

Student t-stest is parametric test, widely applied to evaluate weather the means of two samples have statistically significant difference.<br>
In terms of hypothesis testing we can assume a case when we want to compare, for instance, two groups time consumption of particular webpage. Hence t-test can be performed for continuous outcome. 

To recall the key concepts, the brief explanation of t-test types is formulated below.<br>


1. The **one-sample t-test**: compare the mean of a population with a theoretical value.<br>

 - $t=\frac{m-\mu}{s/\sqrt{n}}$
 - $df=n-1$,<br>
 
where $m$ - population mean, $\mu$ -  theoretical value, $df$ - degrees of freedom.


2. **Unpaired two sample t-test**: compare unrelated groups means (groups sourced from independent samples). 
 - $t= \frac{m_A-m_B}{\sqrt{\frac{S^2}{n_A} + \frac{S^2}{n_B}}}$
 - $S^2=\frac{\sum (x - m_A)^2+\sum (x - m_B)^2}{n_A + n_B - 2}$
 - $df=n_A+n_B-2$,

where $S^2$ - common variance of two samples, $m_A$, $m_B$ - means of groups $A$ and $B$, $n_A$, $n_B$ - sample sizes of $A$ and $B$ groups.


3. **Paired t-test formula**: compare related groups means (groups sourced from same sample).
 - $t = \frac{m}{s/\sqrt{n}}$
 - $df=n-1$
 
 #### t-test: Python implementation
 
Below we are going to perform two sample t-test for random distributed values with respect of N sample size. Assume two groups have same standard deviation and different sample mean, we will calculate t-statistic with related p-value.
<details><summary>Python code</summary> 
  
<p>
  
 ```python
"""
- import visualization libraries
- import stats module to calculate statistics 

"""

from scipy import stats
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
%matplotlib inline

plt.style.use('bmh')

"""
- generate random numbers from sample distribution
- calculate variance and mean to apply formulas 
- use scipy.stats.t and .cdf() function to generate cumulative density function of Student’s t-distribution 
- plot each group distribution with respect of n size

"""    
# initialize n-size list     
N = [10, 30, 50, 100]

for n in N:
    x1 = np.random.randn(n) + 2 # set uз mean difference
    x2 = np.random.randn(n)
    # set up ddof=1 to make variance unbiased 
    var_1 = x1.var(ddof=1)
    var_2 = x2.var(ddof=1)
    mean_1 = x1.mean()
    mean_2 = x2.mean()
    
    diff = mean_1 - mean_2
    s_pol = np.sqrt((var_1+var_2)/2.0)
    # t-statistics calculation based on mean and variance
    t_stat = diff/(s_pol * np.sqrt(2.0/n)) 
    # compute degrees of freedom
    d_f = 2.0*n - 2.0
    # estimate p-value
    p = 1.0 - stats.t.cdf(t_stat, df=d_f)
    p_value = 2.0 * p
    # display t-stat and p-value for each N
    fig, ax = plt.subplots(1,1, sharex=True)
    print 't-statistic = '+str(round(t_stat,2)), 
    print 'p-value=' + '{:.15f}'.format(round(p_value, 10))
    # show distributions
    sns.distplot(x1, kde=True, color="b", label='Group 1')
    sns.distplot(x2, kde=True, color="g", kde_kws={"shade": True}, label = 'Group 2')
    fig.suptitle('Groups density, N size= '+str(n))
    
    plt.legend()
    fig.show()
 ```
 
 </p>
</details>

 ```python
t-statistic = 6.5 p-value=0.000004080900000
t-statistic = 7.33 p-value=0.000000000800000
t-statistic = 10.65 p-value=0.00000000000000
t-statistic = 14.12 p-value=0.00000000000000
 ```
![LSTM]({{ 'AB_test_output/mean_diff_10.png' | absolute_url }})
![LSTM]({{ 'AB_test_output/mean_diff_30.png' | absolute_url }})
![LSTM]({{ 'AB_test_output/mean_diff_50.png' | absolute_url }})
![LSTM]({{ 'AB_test_output/mean_diff_100.png' | absolute_url }})

Assuming 5% signifinace level, we see the groups difference tends to increace as sample size grows.

#### $\chi$-squared statistic for categorical data

While t-test refers to population mean, $\chi$-squared test is commonly used when we deal with categorical data (yes/no, 1/0) and test whether the number of data points is consistant with null hypothesis. <br>
$\chi^2$-statistic reffers to $\chi^2$ distribution and can never possibly be negative: more precisely,  $\chi^2$ test statistic approaches  $\chi^2$ distribution with N=$\infty$ - same as in Central limit theorem case. 

Analysing clicks through rate or conversion rate we look at binary choice and able to apply chi-sqared test, forming the number of successes or failures into *contingency table*: for instance, we test whether the difference between advertisement A and B is statistically significant based on users' clicks. 

Chi-squared statistic can be derived by formula:

$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$, <br>

where $O_i$ - observed value, $E_i$ - expected value.

#### $\chi$-squared in practice 

On the next stage we will see how chi-sqaured test works with binary data.<br>
Creating contingency table based on random choice with 1 or 0 outcome, display the p-value change for different sample size.

<details><summary>Python code</summary> 
  
<p>
  
 ```python
# import chi^2 from scipy library 
import scipy
from scipy.stats import chi2

"""
- create 2x2 matrix as contingency table
- randomly generate A nad B groups with N=3000 and binary outcome
- assume probability difference for 1 and 0 choice between groups
- use  chi2_contingency() from stats module to estimate chi-squared statistic

"""

T = np.zeros((2, 2)).astype(np.float32)
p_value=[]
for i in range(3000):
    A = np.random.choice(np.arange(0, 2), p=[0.4, 0.6])
    B = np.random.choice(np.arange(0, 2), p=[0.45, 0.55])
    T[0][0] += A!=0
    T[0][1] += A!=1
    T[1][0] += B!=0
    T[1][1] += B!=1
    if i>10:
        c2=scipy.stats.chi2_contingency(T, correction=False)
        p_value.append(c2[1])
        
# plot p-value change against 5% significance level 
fig, ax = plt.subplots(figsize=(9,7))
plt.plot(p_value, linewidth=1.5)
plt.plot(np.ones(3000)*0.05, '--', linewidth=1.5)
plt.title('P-value change for N samples', fontsize=14)
ax.set_xlabel('N size', fontsize=12)
plt.show()
 ```
 
 </p>
</details>

![LSTM]({{ 'AB_test_output/chi_square.png' | absolute_url }})

It can be noticed the p-value remains above significant level (5%) till particular N sample size reached and never return.



