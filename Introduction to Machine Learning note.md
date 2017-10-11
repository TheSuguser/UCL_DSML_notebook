# Introduction to Machine Learning

## Week 2

### Probability Refresher

* Joint, marginal, conditional probability

  $X\in \{x+i\}$  and $Y \in \{y_i\}$ 

  $n_{i,}j = \# \{X=x_i \land y = y_j\}$

  $c_j= \# \{X=x_i\}$

  $r_j= \# \{Y=y_i\}$

  Then we can drive

  * Joint probability	 $p(X = x_i, Y=y_i)=\frac{n_{ij}}{N}$
  * Marginal probability     $p(X=x_i)=\frac{c_i}{N}$
  * Conditional probability    $p(Y= y_i|X=x_i)=\frac{n_{ij}}{c_i}$ 

* Continuous Variables

* Gaussian distribution

  * 1-D case
    * Mean $\mu$
    * Variance $\sigma^2$
  * Multi-dimensional case
    * Mean $\mu$
    * Covariance $\sum$ (协方差)

* Likehood of $\theta$

  Probaility that the data $X$ have indeed been generated from a probability density with parameters $\theta$ 
  $$
  L(\theta) = p(X|\theta)
  $$
  For Gaussian distribution: $\theta = (\mu,\sigma)$

  * Computation of the likehood

    * Single data point: $p(x|\theta)$

    * Assumption: all data points are independent :

    * $$
      L(\theta) = p(X|\theta) = \prod_{n=1}^{N}p(x_n|\theta)
      $$

    * Negative (of) log-likelihood
      $$
      E(\theta) = -\mbox{ln}L(\theta) = - \sum^N_{n=1}\mbox{ln}p(x_n|\theta)
      $$

    * Estimation of the likehood $\theta$ (learning)

      * Maximize the likehood
      * Minimize the negative log-likehood

* Bernoulli distribution

  * Parametric model for posterior (尾部)

    $P(Y=1|X=x;w) = f(x,w)$

    $P(Y=0|X=x;w) = 1- f(x,w)$

    $P(Y=y|X=x;w) = f(x,w)^y(1-f(x,w))^{1-y}$

* Bayes' rule

  $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

* Sum and product rule

  * Sum rule:  $P(X)=\sum_YP(X,Y)$
  * Product rule:     $p(X,Y) = P(Y|X)P(X)$

### Logistic Regression

As what we have mentioned before $P(Y=1|X=x;w) = f(x,w)$

Particular chioce of form of $f$:
$$
P(Y=1|X=x;w) = g(w^\top x)
$$
Sigmoidal: $g(\alpha) = \frac{1}{1+\mbox{exp}(-\alpha)}$

Loss function of classification:

$L(S,w) =- \mbox{log}P(y|X;w) = -\sum_{i=1}^N\mbox{log}P(y_i|x_i,w) =\sum_{i=1}^{N}l(y_i,f_w(x_i))$

where 

quadratic loss:    $l(y,f_w(x) )= (y-f_w(x))^2$

* Rewriting the cross-entropy loss

  $h_w(x)=w^\top x$

  $y_\pm = 2y_b-1$ ($y_b\in\{0,1\}$, $y_\pm \in \{-1,1\}$)

  $P(Y=1|X=x;w)=\frac{1}{1+\mbox{exp}(-h_w(x))}$

  $P(Y=-1|X=x;w)=1-P(Y=1|X=x;w）= \frac{1}{1+\mbox{exp}(h_w(x))}$

  Thus,

  $P(Y=y|X=x;w)=\frac{1}{1+\mbox{exp}(-yh_w(x))}$

  $L(S,w)=\sum_{i=1}^{N}\mbox{log}(1+\mbox{exp}(-y_ih_w(x_i)))$

  where

  log loss: $l(y,f(x)) = \mbox{log}(1+\mbox{exp}(-yf(x)))$

  ​

  ​

