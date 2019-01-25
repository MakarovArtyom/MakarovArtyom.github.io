---
title: "Traditional and Bayesian approaches to A/B testing"
date: 2018-10-11
tags: [bayesian statistics, a/b test, hypothesis testing]
header:
  image: "/AB_test_output/ab.jpeg"
excerpt: "Bayesian Statistics, A/B test, Hypothesis testing"
mathjax: "true"
---

# Traditional and Bayesian approaches to A/B testing

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

To recall the key concepts, the brief explanation of t-test types is formulated below.

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
 
 
