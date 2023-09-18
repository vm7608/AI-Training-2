# **Conditional GAN (cGAN)**

## **1. Problem of GANs/DCGANs**

- GANs/DCGANs generate random images. We cannot control the output of the generator.
- We cannot generate images of a specific class.

- **Solution**: Conditional GANs (cGANs) that will help to generate images of a specific class based on additional information (labels y). y can be any additional information such as class labels, image descriptions, etc. and it can be seem as a condition to generate images so we call it conditional GANs (cGANs).

## **2. cGANs architecture**

### **2.1. Generator**

![Generator](https://phamdinhkhanh.github.io/assets/images/20200809_ConditionalGAN/pic1.jpg)

- Generator takes a random noise vector z and a label y as input and produces an image x.

- First, noise vector z is pass through a fully-connected layer and be reshaped into 3D tensor (size 7x7x128 as above image).

- As the same time. labels y are being embedded by one-hot encoding and then reshaped into 3D tensor (size 7x7x1 as above image).

- Then, the noise vector z and the label y are concatenated on the channel dimension (size 7x7x129 as above image) and passed through a series of `ConvTranspose2d` layers to produce an image labeled y.

### **2.2. Discriminator**

![Discriminator](https://miro.medium.com/max/700/1*FpiLozEcc6-8RyiSTHjjIw.png)

- Discriminator is still a binary classification model with task is to classify whether an image is real or fake.

- The real image come from the dataset and the fake image come from the generator. The ratio of real and fake images is 1:1.

- In cGANs, the discriminator takes both the image x and the label y as input and produces a single scalar output.

## **3. Loss function**

- The loss function of cGANs is the same as GANs.

```math
\min_{G} \max_{D} V(D, G) = \underbrace{\mathbb{E}_{x \sim p_{data}(x)} [\log D(x)]}_{\text{log-probability that D predict x is real}} + \underbrace{\mathbb{E}_{z \sim p_{z}(z)} [\log (1-D(G(z)))]}_{\text{log-probability D predicts G(z) is fake}}
```
