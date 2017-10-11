# Introduction to Deep Learning

## Introduction

### Application

* Imagine Classification
* Machine Translation
* Audio Understanding
* Speech Synthesis
* Game Play

### What is a Deep Neural Network?

'**Deep learning**' means using a neural network with serveral layer of nodes between input and output

![](/Users/thesuguser/Desktop/UCL_course/UCL_DSML_notebook/Figure/neural network node.jpg)





![](/Users/thesuguser/Desktop/UCL_course/UCL_DSML_notebook/Figure/neural network structure.jpg)

* Different connections lead to different network structure. Each neurons can have different values of weights and biases. Weights and biases are **network parameters $\theta$ **

* Activation Functions

  * Sigmoid: $\sigma(x)=\frac{1}{1+e^{-x}}$
  * tanh: $tanh(x)$
  * ReLU: $max(0,x)$
  * Leaky ReLU: $max(0.1x,x)$
  * Maxout: $max(w_1^Tx+b_1, w_2^T+b_2)$
  * ELU: $x, \mbox{if }x\geq0;\alpha(e^x-1),\mbox{if }x<0$

* Importance of Activations:

  Adding **nonlinearity** to the network ability to model data.

### Convolutional Neural Network (CNN)

* CNN inspired by the Visual Cortex(视觉皮质).
* CNNs are deep nets that are used for image, object, and even speech recognition.
* Deep supervised neural networks are generally too difficult to train.
* CNNs have multiple types of layers, **the first of which is the convolutional layer**.

![](/Users/thesuguser/Desktop/UCL_course/UCL_DSML_notebook/Figure/CNN.jpg)

### Recurrent(周期) Neural Network (RNN)

* RNNS have a feedback loop where the net's output is fed back into the net along with the next input
* RNNs receive an input and produce an output. Unlike others nets, the inputs and outputs can come in a sequence.
* Variant of RNN is Long Term Short Memory(LSTM)

![](/Users/thesuguser/Desktop/UCL_course/UCL_DSML_notebook/Figure/RNN.jpg)

## Week 2: Unsupervised Methods

### Neural networks

Everything with > 1 hidden layer is **"deep"**

For a single training exmaple $()$