# SNLP note

## Introduction

### Definition

- Building computer systems that **understand** and **generate** natural languages.

- Deep understanding of broad language.

or

- A collection of problems, techniques, ideas, frameworks, etc. that really are not tied together in any resonable way other than the fact that they have to do with NLP

### Some Applications:

- Speech recognition

- Machine translation

- Personal assistants

- Information Extraction

- Summarization

- Generation

- Question Answering

- Sentiment analysis

- Machine Comprehension

- Cognitive Science and Psycholinguistics (认知科学与语言心理学) 

### Syllabus

- Structured prediction

- Preprocessing

- Generative learning

- Discriminative learning

- Weak supervision

- Representation and deep learning

### NLP Tasks

- Tokenization, Segmentation
- Language modeling
- Machine translation
- Syntactic parsing (语法分析)
- Document classification
- information Extraction
- Textual entailment/Machine comprehension (文字蕴含，机器理解)

## Structure Prediction

### Problem Signature

- Given some input structure $x \in X $ , such as a token, sentence, or documents….
- Predic an **output structure** $y \in Y$, such as a class label, a sentence or syntactic(句法) tree.

### Recipe 1: Learn to Score

- Define a prametrized model $s_\theta (x,y)$ that measures the _match_ of a given x and y using _representations_ $f(x)$ and $g(y)$.

- Learn the paramenters $\theta$ from the trainning data $D$ to minimise a loss.

- Given an input x find the highest-scoring output structure
  $$
  y^\star = argmax_{y \in Y}s_\theta (x,y)
  $$
  (a discrere optimization problem)

#### How to estimate $\theta$

Let us define a **Loss Function**
$$
l(\theta) = \sum_{(x,y)\in D} I(y\neq y_\theta^\star(x))
$$
where

- $I(True) = 1$ and $I(False) = 0 $

- $y_\theta^\star(x)\in Y$ is highest scoring translation of x
  $$
  y^\star _\theta(x) = argmax_{y \in Y}s_\theta (x,y)
  $$
  ​

**Learning** is as simple as choosing the parameter with the lowest loss
$$
\theta^\star = argmin_{\theta \in [0,2]}l(\theta)
$$

### Background Reading

* Noah Smith, [Linguistic Structure Prediction](http://www.cs.cmu.edu/~nasmith/LSP/)
    * Free when logging in through UCL 
    * Relevant: 
        * Introduction
        * Dynamic Programming 
        * Generative Models (and unsupervised generative models)
        * Globally Normalized Conditional Log-Linear Models  


## Language Model

Language models calculate the probability to see a given sequence of words, as defined through a tokenization algorithm, in a given language or sub-language/domain/genre.

A language model is a stochastic process that models the probability $p(w_1,….,w_d)$ of observing sequences of words $w_1,…,w_d$ . We can, without loss of generality, decompose (分解) the probability of such sequences into
$$
p(w_1,...,w_d) = p(w_1)\prod_{i=2}^d p(w_i|w1,...,w_{i-1})
$$
This means that a language model can be defined by how it models the conditinal probability  $p(w_i|w1,...,w_{i-1})$ of seeing a word $w_i$ after having seen the *history* of previous words $w_1,…,w_{i-1}$.

### N-gram Language Models

The most common type of equivalence class relies on truncating(缩短) histories $w_1,…,w_{i-1}$ to length $n-1$:
$$
p(w_i|w1,...,w_{i-1}) = p(w_i|w_{i-n},...,w_{i-1})
$$
That is, the probability of a word only depends on the last $n-1$ previous words. 

### A Uniform Baseline(基线，底线) LM

**Unigram models** are the simplest 1-gram language models. That is, they model the conditinal probability of word using the prior probability of seeing that word:
$$
p(w_i|w1,...,w_{i-1})=p(w_i)
$$
To setup dataset and as baseline for more complex language models, we first introduce the simplest instantiation (实例) of a unigram model: a uniform language model which assigns the same prior probability to each word. That is given a vocabulary(词汇表) of words V, the uniform LM is defined as:
$$
p(w_i|w_1,...,w_{i-1})=\frac{1}{|V|}
$$

### Shannon Game

Shannon game: predict what the next word in acutual contxt should be, and win if your predictions match the words in an actual corpus. This can be formalized using the notion of **Perplexity** of the LM on a given dataset. Given a test sequence $w_1,…,w_T$ of $T$ words, we calculate the perplexity $PP$ as follows:
$$
PP(w_1,...,w_T) = p(w_1,...,w_T)^{-\frac{1}{T}} = \sqrt[T]{\prod^T_i\frac{1}{p(w_i|w_{i-n},...,w_{i-1})}}
$$

###Out-of-Vocabulary Words

The baseline model assign zero probability to words that are not in the vocabulary. Test sets will usually contatin such words, and this leads to the infinite perplexity. 

### The Long Tail

Generally we will see few words that appear repeatedly, and a long tail of words that appear only a few times. While each individual long-tail word is rare, the probability of seeing any long-tail word is quite high (the long tail covers a lot of the frequency mass)

#### Zipf's law

Let $r_w$ be the rank of a word $w$, and $f_w$ its frequency, then we have
$$
f_w \propto \frac{1}{r_w}
$$

### Inserting Out-of-Vocabulary Tokens

One of methods that generally simplifies the handling of unseen words, and gives rise to a simple smoothing heuristic (启发式算法?) is that we replace unseen words in the test corpus with an out-of-vacabulary token, say `OOV` . This means that LMs can still work with a fixed vocabularly that  consists of all training words, and the `OVV` token to avoid the infinite perplexity problem.

### Training Language Models

To improve upon the baseline, we can estimate the conditional n-gram distributions from the tranning data. To this end let us first introduce one parameter $\theta_{w,h}$ for each word $w$ and history $h$ of length $n-1$, and deffine a parametrized language model $p_\theta$:
$$
p_\theta(w|h)=\theta_{w,h}
$$
Training an n-gram LM amounts to estimating $\theta$ from some training set $D=(w_1,…,w_n)$. One way to do this is to choose the $\theta$ that maximizes the log-likehood of $D$:
$$
\theta^\star = \mbox{argmax}_\theta \mbox{log }p_\theta(D)
$$
As it turns out, this maximum-log-likelihood estimate(MLE) can calculated in closed form, simply by counting:
$$
\theta_{w,h}^\star = \frac{\# _D(w,h)}{\#_D(h)}
$$
where 
$$
\#_D(e) =\mbox{ Count of envent }e\mbox{  in }D
$$
Here the event $h$ means seeing the history $h$, and $w$, $h$ seeing the history $h$ followered by word $w$.

