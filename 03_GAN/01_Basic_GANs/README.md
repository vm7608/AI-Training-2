# **Generative Adversarial Networks (GANs)**

## **1. Overview and introduction**

### **1.1. What is GANs?**

- Generative Adversarial Networks (GANs) were developed in 2014 by Ian Goodfellow and his teammates. GAN is basically a generative model that generates a new set of data based on training data that look like training data.

- To understand the term GAN let’s break it into separate three parts:
  - Generative – To learn a generative model, which can captures the distribution of training data and can be used to generate new samples.
  - Adversarial – The training of the model is done in an adversarial setting.
  - Networks – use deep neural networks for training purposes.

- GANs have two main blocks (two neural networks) which compete with each other. They are Generator and Discriminator. The generator tries to generate new data instances while the discriminator tries to distinguish between real and fake data.

### **1.2. Discriminative vs Generative models**

- Machine learning models can be broadly classified into two categories: discriminative and generative models (as one of ways to classify ML models).

![Discriminative vs Generative models](https://imgur.com/xjtCWqw.png)

- Discriminative models base on input features x to predict label or value of output y (e.g. classification, regression). The predicted value is a conditional probability based on input features: `P(y|x)`. For example in binary classification:

```math
p(y|\mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^\intercal\mathbf{x}}}
```

- Generative models, on the other hand, try to learn the joint probability distribution `P(x|y)` of the input features x and the label y. The models will concentrate on finding what is the input features properties when we already know the label y. Generative models based on `Bayes theorem`:

```math
P(x|y) = \frac{P(y|x)P(x)}{P(y)}
```

- For example, we have a dataset of bad dept with 2 input features x1, x2 and 1 output label y (0: Non-Fraud, 1: Fraud).

![Bad dept dataset](https://imgur.com/dWC2zfO.png)

- In the left, discriminative model will try to find the boundary between 2 classes (Non-Fraud and Fraud) based on input features x1, x2.
- In the right, generative model will try to find the probability distribution of input features x1, x2 when we already know the label y (Non-Fraud or Fraud). Based on that, it can generate new samples that look like the training data (blue square). Note that, with generative models, we need to know the label y of data.

### **1.3. Types of Generative models**

- Explicit model: try to find probability distribution of input features x base on a pre-assumed probability distribution function of the input. To generate new samples, we just need to sample from the probability distribution function of the input.

- Implicit model: a simulator model that can generate new samples that look like the training data. New samples are generated directly from the model without any pre-assumed probability distribution function of the input.

![Types of Generative models](https://imgur.com/MelJzGj.png)

## **2. GANs architecture**

### **2.1. GANs intuition**

- GANs intuition based on `zero-sum non-cooperative game`.
  - In this game, there are two players and there will be one winner and one loser.
  - In each turn, both of them need to maximize their own utility (or minimize their own loss).
  - At a time, the game will reach a equilibrium point where neither of them can improve their utility (or loss) by changing their strategy. We call that point `Nash equilibrium`.

- The generator network takes random input (noise) and generates samples, such as images, text, or audio that look same as the training data it was trained on. The goal of the generator is to produce samples that are indistinguishable from real data.

- The discriminator network tries to distinguish between real and generated samples. It is trained with real samples from the training data and generated samples from the generator. Its objective is to correctly classify real data as real and generated data as fake.

- The training process is an adversarial game between the generator and the discriminator.
  - The generator aims to produce more realistic samples that fool the discriminator.
  - The discriminator tries to improve its ability to distinguish between real and generated data.
  - This adversarial training pushes both networks to improve over time.
  - Ideally, this process converges to a point where the generator is capable of generating high-quality samples that are difficult for the discriminator to distinguish from real data (Nash equilibrium).

![GANs intuition](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/11/d_rk.png?resize=768%2C357&ssl=1)

![Fake money example](https://dz2cdn1.dzone.com/storage/temp/10276123-dzone.png)

### **2.2. Generator**

![Generator architecture](https://imgur.com/a4p9G3d.png)

- In basic, Generator is a neural network that takes random input (noise) and generates samples, such as images, text, or audio that look same as the training data it was trained on.

- Input noise is intialized ramdomly from a normal distribution (Gaussian distribution) or uniform distribution. In some modern GANs, input can be images, text, or audio. But in original GANs paper, input is random noise.

- From the input ramdom noise z, generater is a deep neural network that generates new samples that look like the training data. This is done by transforming the input noise z into a sample that has the same shape as the training data. Then these fake samples are fed into the Discriminator.

### **2.3. Discriminator**

![Discriminator architecture](https://imgur.com/vGjX6DM.png)

- Discriminator is a neural network that tries to distinguish between real and generated samples. It is trained with real samples from the training data and generated samples from the generator. Its objective is to correctly classify real data as real and generated data as fake.

- Label is real if input data is real data from training data. Label is fake if input data is generated data from generator.

- It's simply a binary classifier.

### **2.4. Loss function**

#### **2.4.1. Review cross entropy loss**

- For binary classification problems, we have the loss function for a `single training` instance:

```math
c(\theta) = \begin{cases}
-log(\hat{p}) & \text{if } y = 1, \\
-log(1 - \hat{p}) & \text{if } y = 0.
\end{cases}
```

- This loss function makes sense because:
  - $`-log(t)`$ grows very large when $`t`$ approaches 0 and approaches 0 when $`t`$ approaches 1.
  - The cost will be large if the model make a wrong prediction
    - $`y = 0`$ but model estimates $`\hat{p}`$ close to 1
    - $`y = 1`$ but model estimates $`\hat{p}`$ close to 0
  - The cost will be close to 0 if the model makes a right prediction

- The cost function over `the whole training set` is simply the average cost over all training instances (called the **log loss**):

```math
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y_i log(\hat{p}_i) + (1 - y_i) log(1 - \hat{p}_i)]
```

- This cost function is convex, so Gradient Descent (or any other optimization algorithm) is guaranteed to find the global minimum (if the learning rate is not too large and you wait long enough).

![Log Function](https://raw.githubusercontent.com/shruti-jadon/Data_Science_Images/main/cross_entropy.png)

#### **2.4.2. Discriminator loss**

- The discriminator is a binary classifier to distinguish if the input $`x`$ is real (from real data) or fake (from the generator).

- Typically, the discriminator outputs a scalar prediction $`o\in\mathbb R`$ for each input $`x`$, such as using a fully connected layer with hidden size 1, and then applies sigmoid function to obtain the predicted probability:

```math
D(\mathbf x) = \frac{1}{1+e^{-o}}
```

- Assume the label y for the true data is 1 and 0 for the fake data. We train the discriminator to minimize the cross-entropy loss, i.e.,

```math
\min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \}
```

#### **2.4.3. Generator loss**

- For the generator, it first draws some parameter $`\mathbf z\in\mathbb R^d`$ from a source of randomness, e.g., a normal distribution $`\mathbf z \sim \mathcal{N} (0, 1)`$. We often call z as the latent variable.

- It then applies a function to generate $`\mathbf x'=G(\mathbf z)`$. The goal of the generator is to fool the discriminator to classify $`\mathbf x'=G(\mathbf z)`$ as true data, i.e., we want $`D( G(\mathbf z)) \approx 1`$.

- In other words, for a given discriminator D, we update the parameters of the generator G to maximize the cross-entropy loss when y = 0.

```math
\max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.
```

- If the generator does a perfect job, then $`D(\mathbf x')\approx 1`$  so the above loss is near 0, which results in the gradients that are too small to make good progress for the discriminator. So commonly, we minimize the following loss, which is just feeding $`\mathbf x'=G(\mathbf z)`$ into the discriminator but giving label y = 1.

```math
\min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \},
```
  
- To sum up, D and G are playing a “minimax” game with the comprehensive objective function:

```math
\min_D \max_G \{ -E_{x \sim \textrm{Data}} \log D(\mathbf x) - E_{z \sim \textrm{Noise}} \log(1 - D(G(\mathbf z))) \}.
```

#### **2.4.4. GANs loss function (min-max GANs loss)**

### **2.5. Training process**

## **3. Applications and challenges of GANs**

## **4. GANs variants overview**

### **4.1. Conditional GANs**

### **4.2. Deep Convolutional GANs (DCGANs)**

### **4.3. StyleGANs**

### **4.4. CycleGANs**
