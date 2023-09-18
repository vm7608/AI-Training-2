# **Deep Convolutional Generative Adversarial Networks (DCGANs)**

## **1. Introduction**

- The main change in DCGANs is the use of convolutional and convolutional-transpose layers instead of fully-connected layers.
- The generator uses `ConvTranspose2d` layers to produce an image from a random noise vector.
- The discriminator uses `Conv2d` layers to produce a single scalar output.

## **2. DCGANs**

![Architecture](https://editor.analyticsvidhya.com/uploads/2665314.png)

- Here is the summary of change in the architecture in DC:
  - Replace all max pooling with convolutional stride
  - Use transposed convolution for upsampling.
  - Eliminate fully connected layers.
  - Use Batch normalization except the output layer for the generator and the input layer of the discriminator.
  - Use ReLU in the generator except for the output which uses tanh. Ouput layer in generator use tanh to produce an image in the range [-1, 1]. (Why? Because the input image is normalized to [-1, 1] and tanh is the only activation function that can produce negative values. Moreover, tanh is a smooth function which is good for backpropagation.)
  - Use LeakyReLU in the discriminator.

- Loss function of DC-GANs is the same as GANs.
