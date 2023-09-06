# **Transfer Learning**

## **1. What is Transfer Learning?**

- Situation:
  - You have a small dataset that cant be used to train a deep learning model from scratch.
  - There exists a model that was trained on a similar task to the one you want to solve.
  - Solution: Use the pre-trained model as a starting point for your model.

- Definition: Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

- Transfer learning only works in deep learning if the model features learned from the first task are general.

- "In transfer learning, we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task. This process will tend to work if the features are general, meaning suitable to both base and target tasks, instead of specific to the base task."

- This form of transfer learning used in deep learning is called inductive transfer. This is where the scope of possible models (model bias) is narrowed in a beneficial way by using a model fit on a different but related task.

![Transfer Learning](https://machinelearningmastery.com/wp-content/uploads/2017/09/Depiction-of-Inductive-Transfer.png)

## **2. How to Use Transfer Learning?**

- Two common approaches of using Transfer Learning:
  - Use the pre-trained model as a fixed feature extractor
  - Fine-tune the pre-trained model

### **2.1. Feature extractor**

- The pre-trained model is used as a feature extractor for the dataset of interest.
- After getting the output from the pre-trained model, we can train a new model with these features as input. This new model is usually a simple model like SVM, Logistic Regression, etc.

- For example of using a VGG16 pre-trained model as a feature extractor:
  - We remove the last fully connected layer of the VGG16 model. Now, VGG16 model is a feature extractor, it takes an image as input and outputs features.
  - We train a new model (SVM, Logistic Regression, etc.) on these features for our own task.

### **2.2. Fine-tuning**

- We remove the old fully connected layers of the pre-trained model and replace them with new fully connected layers with random weights.

- We have 2 periods of training:
  - First period:
    - Because the weights of new fully connected layers are random while the weights of the pre-trained model are good, we freeze the weights of the pre-trained model and train the new fully connected layers first.
    - We train the new fully connected layers until the model learns something and move to the second period.
  - Second period:
    - We unfreeze the weights of the pre-trained model and train the whole model.
    - We can unfreeze all the layers of the pre-trained model or just some of them based on time and resources.

- For example of fine-tuning a VGG16 pre-trained model:

|Add new layer|First period|Second period|
|---|---|--|
|![1](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/04/fine-tune.png?w=489&ssl=1)|![2](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/04/freeze_part.png?w=446&ssl=1)|![3](https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/04/unfreeze_all.png?w=405&ssl=1)|

## **3. When to Use Transfer Learning?**

- Transfer learning is an optimization, a shortcut to saving time or getting better performance.

- 3 benefits of using transfer learning:
  - Higher start: The initial skill (before refining the model) on the source model is higher than it otherwise would be.
  - Higher slope: The rate of improvement of skill during training of the source model is steeper than it otherwise would be.
  - Higher asymptote: The converged skill of the trained model is better than it otherwise would be.

![Transfer Learning](https://machinelearningmastery.com/wp-content/uploads/2017/09/Three-ways-in-which-transfer-might-improve-learning.png)

- There are 2 most important factor of using Transfer Learning:
  - Dataset size: Small or Large
  - Similarity of the problem being solved: Similar or Different

- There are 4 cases of using Transfer Learning:
  - **Small Dataset, Similar Problem:** Use the pre-trained model as a fixed feature extractor. (Because our data is small so when using fine-tuning, the model can be overfitting. Moreover, the problem is similar so the features learned from the pre-trained model are general.)

  - **Small Dataset, Different Problem:** Because the data is small, we should use a feature extractor to avoid overfitting. However, because the data are different, we should not use the feature extractor with the entire ConvNet of the pre-trained model, but only use the first layers. The reason is because the early layers will learn more general features, while later layers will learn more specific features.

  - **Large Dataset, Similar Problem:** Fine-tune the pre-trained model. (Because the data is large, we can fine-tune the pre-trained model without overfitting)
  - **Large Dataset, Different Problem:** We can train the model from sratch, but it will be better if we initialize the weights of the model with the weights of the pre-trained model.

## **4. Notice**

- Because the pre-trained model is trained on a different dataset, the input of the pre-trained model is different from the input of our model. Therefore, we need to do some preprocessing to the input of the pre-trained model to make it suitable for our model.

- We should use a smaller learning rate for the pre-trained model than the learning rate of the new layers. Because the pre-trained model is already trained, we don't want to change its weights too much. We just want to change the weights of the new layers.
