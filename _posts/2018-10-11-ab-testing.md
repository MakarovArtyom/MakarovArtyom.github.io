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
Generally, this approach assumes the parameters of distribution are fixed and data generated randomly with respect of maximum likelihood function: 
$$\hat\theta = argmax_\theta(X |\theta)$$

We consider mean that stands for "bellshape" center and standard deviation (width) as parametres of normal distribution. <br>
The probability in terms of frequentist statistics measures as the long-term frequency of event occurrence. However we don't actually know the real mean value - we know it's fixed and can be estimated from population sample only. <br>
Common techniques, that frequentists use to make an assumption about estimate are:<br>

- $H_0$ hypothesis formulation;
- Data collection;
- Test statistics calculation with respect of p-values;
- Confidence interval.

Theoretically we need to collect an infinite number of samples from population and constract CI (for istance, 95% probability) for each one, but in practice we draw a single sample and expect 95% of the interval estimates contain population parameter.
