# Supervised Learning 

## Week 1

### Supervised Learning Problem

Given a set of **input/output** pairs (**tranning set**) we wish to compute the functional relationship between the input and the output.

### Learning approach

* Stable: finds something that is not chance part of set of examples.
* Efficient: infers(推断) solution in time polynomial in the size of the data
* Robust: should not be too sensitive to mislabelled/noisy examples

### Supervised Learning Model

* Goal: Given training data(pattern, target) pairs
  $$
  \mathcal{S} = \left\{(x_1,y_1),...,(x_m,y_m)\right\}
  $$
  infer a function $f_s$ such that
  $$
  f_\mathcal{S}(x_i)\approx y_i
  $$
  for the future data
  $$
  S' = \{(x_{m+1},y_{m+1}),(x_{m+2},y_{x+2},...\}
  $$

* $\mathcal{X}$ : input space (eg, $\mathcal{X} \subseteq \mathbb{R}^d $), with elements $x,x',x_i,…$

* $\mathcal{Y}$ : output space, with elements $y,y',y_i,…$

### Learning Algorithm 

* Traning set: $\mathcal{S}=\{(x_i,y_i)^m_{i=1}\}\subseteq \mathcal{X}\times\mathcal{Y}$
* A **learning algorithm** is a mapping $\mathcal{S} \rightarrow f_\mathcal{S}$
* A new input $x$ is predicted as $f_\mathcal{S}(x)$

### Learning Regression

$$
\mathrm{Xw=y}
$$

the linear predictor
$$
\hat{y}=\mathbf{w\cdot x}
$$
Note: $\mathrm{A\cdot B}=A^TB$

Find a linear predictor $\mathrm{\hat{y}}=\mathbf{w\cdot x}$ to minimize the square error over the data $\mathcal{S} = \left\{(x_1,y_1),...,(x_m,y_m)\right\}$ thus
$$
\mathbf{Minimize}: \sum^m_{i=1}(y_i-\hat{y_i})^2 = \sum_{i=1}^m(y_i-\mathbf{w\cdot x})
$$
Thus in martix notation **empirical**(经验主义的) **mean (sqaure) error** of the linear predictor $\hat{y}=\mathbf{w\cdot x}$ on the data sequence $\mathcal{S}$ is


$$
\varepsilon_{emp}(\mathcal{S},\mathbf{w}) = \frac{1}{m}\sum^m_{i=1}(y_i-\hat{y_i})^2
\\=\frac{1}{m} \sum^m_{i=1}(y_i-\hat{y_i})^2 \\= \sum_{i=1}^m(y_i-\mathbf{w\cdot x})^2\\= \frac{1}{m} \sum^m_{i=1}(y_i-\hat{y_i})^2\\ = \sum_{i=1}^m(y_i-\sum^n_{j=1}w_jx_{i,j})^2 \\= \frac{1}{m}\mathbf{(Xw -y)^\top(Xw-y)}
$$
To compute the minimum we solve for 
$$
\nabla_\mathbf{w}\varepsilon_{emp}(\mathcal{S},\mathbf{w}) = 0
$$
So we conclude that
$$
\mathbf{w}=\mathbf{(X^\top X)^{-1}X^\top y}
$$

### K-nearest neighbours (KNN)

* Algorithm

  Let $N(\mathbf{x};k)$ be the set of k nearest training inputs to $\mathbf{x}$ and

$$
I_\mathbf{x}=\{i:\mathbf{x_i}\in N(\mathbf{x};k)\}
$$

​	 the coresponding index set
$$
f(\mathbf{x})=\begin{cases}red &\mbox{if } \frac{1}{k}\sum_{i\in I_\mathbf{x}}y_i>\frac{1}{2} \\green &\mbox{if }  \frac{1}{k}\sum_{i\in I_\mathbf{x}}y_i\leq\frac{1}{2} 
\end{cases}
$$

### Perspectives(观点，前景) on supervised learning

#### Optimal Supervised Learning

* **Model**：We assume that the data is obatined by sampling from a **fixed but unknown** probability density $P(\mathbf{x},y)$

  Expected error:
  $$
  \varepsilon(f) = E[(y-f(\mathbf{x})^2] = \int (y-f(\mathbf{x}))^2 dP(\mathbf{x},y)
  $$
  Our goal is to minimize $\varepsilon$

* **Optimal solution**： $f^\star := \mbox{argmin}_f\varepsilon(f)$  (Called Bayes estimator)

* Bayes estimator for square loss:

  Let us compute the optimal solution $f^\star$ for regression $\mathcal{Y}=\mathbb{R}$..

  Using the decomposition $P(y,\mathbf{x})=P(y|\mathbf{x})P(\mathbf{x})$, we have
  $$
  \varepsilon(f) = \int_{\mathcal{X}}\{\int_\mathcal{Y}(y-f(\mathbf{x}))^2dP(y|\mathbf{x})\}dP(\mathbf{x})
  $$
  So we may see that $f^\star$ is

  **WHY？**
  $$
  f^\star(x) = \int_\mathcal{Y}ydP(y|\mathbf{x})
  $$
  Deriving $f^\star$ with lighter notation
  $$
  f^\star(x) = \sum_{y\in Y}yp(y|x) = E[y|x]
  $$
  We now additionally assume there exist some underlying function $F$ such that
  $$
  y = F(x) +\epsilon
  $$
  where $\epsilon$ is white noise, i.e., $E[\epsilon] = 0$ and finite variance.

  Thus the optimal prediction is 
  $$
  f^\star(x):=E[y|x] = F(x)
  $$
  with square loss.

  We would like to understand the expected error by an arbitrary learner $A_\mathcal{S}(x)$

  Our goal will be to understand the expected error at $x'$
  $$
  \varepsilon(A(x')) = E[(y'-A(x'))^2]
  $$
  where $y'$ is a sample from $P(Y|x')$
  $$
  E[(y'-A(x'))^2] = E[(y'-f^\star(x'))^2]+\\(f^\star(x)-E[A(x')]^2)+\\E[(A(x')-E[A(x')])^2]
  $$

  * Bayes error: $E[(y'-f^\star(x'))^2]$

    is the irreducible noise

  * Bias: $(f^\star(x)-E[A(x')]^2)$

    describes the discrepancy(差异) between the algorithm and "truth"

  * Variance: $E[(A(x')-E[A(x')])^2]$

    capture the variance of the algorithm between training sets.

* Bias and Variance Dilemma

  * The bias and variance tend to trade off against one another
  * Many parameters better flexibility to fit the data thus low bias but high variance
  * Few parameters give high bias but the fit between different data sets will not change much thus low variance
  * This exact decomposition only holds for the square loss.

### Asymptoitic(渐近的) Optimality of k-NN

* As the number samples goes to infinity the error rate is no more than twice the Bayes error rate.

  **TBC**

### Hypothesis Space

We introduce a **restricted** space of functions $\mathcal{H}$ called **hypothesis space**.

We minimize $\varepsilon_{emp}(\mathcal{S},f)$ with $\mathcal{H}$. That is, our learning algorithm is:
$$
f_\mathcal{S}=\mbox{argmin}_{f\in\mathcal{H}}\varepsilon_{emp}(\mathcal{S},f)
$$
This approach is usually called **empirical error(risk) minimization**

For example (Least Squares):
$$
\mathcal{H} = \{f(\mathbf{x}) = \mathbf{w^\top x:w\in \mathbb{R}}^n\}
$$

#### Summary

* Data $\mathcal{S}$ sampled i.i.d from $P$ (fixed but unknown)
* $f^\star$ is what we want, $f_\mathcal{S}$ is what we get
* Different approaches to attempt to estimate/approximate $f^\star$:
  * Minimize $\varepsilon_{emp}$ in some restricted space of functions (eg, linear)
  * Compute local approximation of $f^\star$ (k-NN)
  * Estimate $P$ and then use Bayes rule...

### Model selection



## Kernals and Regularization

### Future Maps

#### Definition

* A feature map is simply a function that maps the "inputs" into a new space.
* Thus, the origin method is now nonlinear in original "inputs" but linear in the "mapped inputs".

#### Linear interpolation (插值)

A problem is well-posed, if

* a solution exists
* the solution is unique
* the solution depends continuously on the data

Learning problems are in general ill-posed, usually because of (2).

**Regularization theory** provides a general framework to solve ill-posed problems.

### Ridge Regression

Given a set of $k$ hypothesis classes $\{\mathcal{H}_r\}_r\in \mathbb{N}_k$ we can choose an appropriate hypothesis class with **cros-validation** 

**Regularization**: An alternative(选择) compatible(兼容) with linear regression is to choos a single "complex" hypothesisi class and then modify the error function by adding a "complexity" term which penaltizes complex function.

Cross-validation may still be needed to set the regularization parameter and other parameters defining the complexity term.

We minimized the **regularized (penalized)** empirical error
$$
\varepsilon _{\mbox{emp}_\lambda}(\mathbf{w}) := \sum_{i=1}^m(y_i -\mathbf{w}^\top \mathbf{x}_i )^2 + \lambda \sum_{l=1}^n w_l^2\equiv (\mathbf{y-Xw})^\top\mathbf{(y-Xw)}+\lambda\mathbf{w^\top w}
$$
The **postive** parameter $\lambda$ defines a trade-off (折衷) between the error on the data and the norm of the vector $\mathbf{w}$ (**degree of regularization**)

Setting $\nabla \varepsilon _{\mbox{emp}_\lambda}(\mathbf{w})=0$ (**details shown in slides** check your ipad)

It can be shown that the regularized solution can be written as
$$
\mathbf{w} = \sum_{i=1}^{m}\alpha_i\mathbf{x}_i  \rightarrow f(\mathbf{x}) = \sum_{i=1}^m \alpha_i \mathbf{x}_i^\top\mathbf{x}
$$
where the vector of parameters $\mathbf{\alpha} = (\alpha_1,…,\alpha_m)^\top$ is given by
$$
\mathbf{\alpha} = (\mathbf{XX^\top}) + \lambda\mathbf{I}_m)^{-1}\mathbf{y}
$$
We can $f(\mathbf{x})=\mathbf{w}^\top \mathbf{x}$ the **primal form** and  $f(\mathbf{x}) = \sum_{i=1}^m \alpha_i \mathbf{x}_i^\top\mathbf{x}$ the **dual form** .

The **dual form** is computationally convenient when $n>m$ . 

* Training time:

  Solving for $\mathbf{w}$ in primal form requires $O(mn^2+n^3)$ operations while solving for $\alpha$ in the dual form requires $O(nm^2+m^3)$ . If $m\ll n$, it is more efficient to use the dual representation. 

* Test time:

  Computing $f(\mathbf{x})$ on a test vector $\mathbf{x}$ in the primal form requires $O(n)$ operations while the dual form requires $O(mn)$ operations.

### Basis Functions (Explicit(显性) Feature Maps)

By a **feature map** we mean a fuction $\mathbf{\phi}$: $\mathbb{R}^n\rightarrow\mathbb{R}^N$
$$
\mathbf{\phi}(\mathbf{x}) = (\phi_1(\mathbf{x}),...,\phi_N(\mathbf{x}))^\top, \mathbf{x}\in\mathbb{R}
$$
Vector $\mathbf{\phi(x)}$ is called the **feature vector** and the space $\{\mathbf{\phi(x)}\}:\mathbf{x}\in \mathbb{R}^n$ the feature space

The non-linear regression function has the primal representation
$$
f(\mathbf{x}) = \sum_{j=1}^Nw_j\phi_j(\mathbf{x})
$$
More generally for second order correlations if $\mathbf{x}\in \mathbb{R}^n$ we have
$$
\mathbf{\phi(x)}:= (\mathbf{x},x_1x_1,x1_x2,...,x1_xn,x_2x_2,x_2x_3,...,x_2x_n,...,x_nx_n)^\top
$$
i.e., $\phi:\mathbb{R}^n\rightarrow \mathbb{R}^{\frac{n^2+3n}{2}}$

### Kernal Functions (Implicit(隐性) Feature Maps)

Given a feature map $\phi$ we define its associated kernel function

$K: \mathbb{R}^n \times \mathbb{R}^n\rightarrow\mathbb{R}$ as
$$
K(\mathbf{x,t}) = \langle\mathbf{\phi(x),\phi(t)}\rangle, \mathbf{x,t}\in\mathbb{R}^n
$$

#### Positive Semidefinite(半定) Kernel

* **Definition**:

  A function $K$: $\mathbb{R}^n\times\mathbb{R}^n\rightarrow \mathbb{R}$ is **positive semidefinite** if it is symmetric and the matrix ($K(\mathbf{x}_i,\mathbf{x}_j):i,j=1,…,k$) is positive semidefinite for every $k\in\mathbb{N}$ and every $mathbf{x_1,…,x_k}\in\mathbb{R}^2 $ 

  $K$ is positive semidefinite if and only if
  $$
  K(\mathbf{x,t}) = \langle\phi(\mathbf{x}),\mathbf{\phi(t)}\rangle, \mathbf{x,t}\in \mathbb{R}^n
  $$
  for some feature map $\phi:\mathbb{R}^n\rightarrow \mathcal{W}$ and a Hilbert space $\mathcal{W}$



####Two Example Kernels

* Polynomial Kernerls

  if $p$ : $\mathbb{R}\rightarrow\mathbb{R}$ is a polynomial with nonnegative coefficients then $K(\mathbf{x,t}) = p(\mathbf{x^\top t}),\mathbf{x,t}\in \mathbb{R}^n$ is postive semidefinite kernel. For example if $a \geq 0$ 

  *  $K(\mathbf{x,t}) = (\mathbf{x^\top t})^r$
  *  $K(\mathbf{x,t}) = (a+\mathbf{x^\top t})^r$
  *  $K(\mathbf{x,t}) = \sum _{i=0}^d \frac{a^i}{i!}(\mathbf{x^\top t})^i$

  are each positive semidefinite kernels.

* Gaussian Kernel

  An important example of a "radial" kernel is the Gaussian kernel
  $$
  K(\mathbf{x,t}) = \mbox{exp}(-\beta||\mathbf{x-t}||^2), \beta>0,\mathbf{x,t}\in \mathbb{R}^n
  $$
  note: any corresponding feature map $\phi(.)$ is $\infty$-dimentional.

#### Kernel Construction

If $K_1,K_2$ are kernels, $a\geq 0$ , $A$ is a symmetric positive semi-definite matrix, $K$ a kernel on $\mathbb{R}^n$ and $\phi: \mathbb{R^n\rightarrow R^N}$ then the following functions are positive semidefinite kernel on $\mathbb{R^n}$

1. $\mathbf{x}^\top A \mathbf{t}$
2. $K_1(\mathbf{x,t})+K_2 (\mathbf{x,t})$
3. $aK_1(\mathbf{x,t})$
4. $K_1(\mathbf{x,t})K_2(\mathbf{x,t})$
5. $K(\phi(\mathbf{x}),\phi(\mathbf{t}))$



## Online Learning

### Learning with Expert Advice

Goal: Design master algorithms with "small loss".

####Halving(二分) Algorithm

## Week 5 Support Vector Machine

### Optimal separating hyperplane





### Soft margin separation





### Support vector machine



### Connection to regularization






















