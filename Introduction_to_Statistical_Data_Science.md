# Introduction to Statistical Data Science

## Chapter 1

* Notation:

  * Upper case latter will be used to donate random variables	- X, Y, Z, etc
  * Lower case letters will be used to denote values taken by random variables     - x, y, z, etc

* Random variables

* Density functions

* Normal distribution:

  * Mean: $\mu$

  * Variance: $\sigma ^2$ (Standard deviation: $\sigma$
    $$
    p(x) = \frac{1}{\sqrt{2\pi \sigma ^2}}\mbox{exp}\{- \frac{(x-\mu)^2}{2\sigma^2}\}
    $$

  * The mean is the location parameter:

    * It tells you where the "peak" of the density will be located. $p(x)$ is maximised when $x = \sigma$

  * The variance is the scale parameter:

    * The varicance controls the degree to which the probability mass/density "spreads" around the mean.

* Q-Q plot

* Expected value of functions of random variables.

  ​

## Chapter 2: Hypothesis testing and confidence intervals

### Hypothesis tests and p-values

* Statistical hypothesis testing involves a trade-off between two drawbacks
  * False positives: rejecting $H_0$ when it is true
  * False negatives: not rejecting $H_0$ when it is true
  * **Note** that not rejecting $H_0$ is not the same as accepting $H_0$

