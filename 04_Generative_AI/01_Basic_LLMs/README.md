# **LARGE LANGUAGE MODELS**

## **1.Overview and Introduction**

### **1.1. Language Models (LMs)**

A language model is a probabilistic model of a natural language that can generate probabilities of a series of words, based on text corpora in one or multiple languages it was trained on. Language models are useful for a variety of tasks, including speech recognition (helping prevent predictions of low-probability (e.g. nonsense) sequences), machine translation, natural language generation (generating more human-like text), optical character recognition, handwriting recognition, grammar induction,information retrieval and other.

Language modeling (LM) uses statistical and probabilistic techniques to determine the probability of a given sequence of words occurring in a sentence. Hence, a language model is basically a probability distribution over sequences of words:

```math
P(x^{(t+1)} | x^{(t)}, x^{(t-1)}, ..., x^{(1)})
```

Here, the expression computes the conditional probability distribution where $`x^{(t+1)}`$ can be any word in the vocabulary.

Language models generate probabilities by learning from one or more text corpus. A text corpus is a language resource consisting of a large and structured set of texts in one or more languages. Text corpus can contain text in one or multiple languages and is often annotated.

One of the earliest approaches for building a language model is based on the `n-gram`. An `n-gram` is a contiguous sequence of n items from a given text sample. Here, the model assumes that the probability of the next word in a sequence depends only on a fixed-size window of previous words:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Language-Model-N-gram.jpg" >
  <br>
  <i>N-gram</i>
</p>

However, n-gram language models have been largely superseded by `neural language models`. It’s based on neural networks, a computing system inspired by biological neural networks. These models make use of continuous representations or embeddings of words to make their predictions:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Language-Model-Neural-Network.jpg" >
  <br>
  <i>Neural Language Model</i>
</p>

Basically, neural networks represent words distributedly as a non-linear combination of weights. Hence, it can avoid the curse of dimensionality in language modeling. There have been several neural network architectures proposed for language modeling such as Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Transformer.

### **1.2. Large Language Models (LLMs)**

Large Language Models (LLMs) are basically neural language models working at a larger scale. A large language model consists of a neural network with possibly billions of parameters. Moreover, it’s typically trained on vast quantities of unlabeled text, possibly running into hundreds of billions of words.

Large language models also called deep learning models, are usually general-purpose models that excel at a wide range of tasks. They are generally trained on relatively simple tasks, like predicting the next word in a sentence.

However, due to sufficient training on a large set of data and an enormous parameter count, these models can capture much of the syntax and semantics of the human language. Hence, they become capable of finer skills over a wide range of tasks in computational linguistics.

This is quite a departure from the earlier approach in NLP applications, where specialized language models were trained to perform specific tasks. On the contrary, researchers have observed many emergent abilities in the LLMs, abilities that they were never trained for.

For instance, LLMs have been shown to perform multi-step arithmetic, unscramble a word’s letters, and identify offensive content in spoken languages. ChatGPT, a popular chatbot built on top of OpenAPI’s GPT family of LLMs, has cleared professional exams like the US Medical Licensing Exam.

### **1.3. LLMs and Foundation Models**

A foundation model generally refers to `any model trained on broad data that can be adapted to a wide range of downstream tasks`. These models are typically created using deep neural networks and trained using self-supervised learning on many unlabeled data.

The term was coined not long back by the Stanford Institute for Human-Centered Artificial Intelligence (HAI). However, there is no clear distinction between what we call a foundation model and what qualifies as a large language model (LLM).

Nevertheless, LLMs are typically trained on language-related data like text. But a `foundation model is usually trained on multimodal data`, a mix of text, images, audio, etc. More importantly, a foundation model is intended to serve as the basis or foundation for more specific tasks:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Foundation-Models.jpg" >
  <br>
  <i>Foundation Model</i>
</p>

Foundation models are `typically fine-tuned with further training for various downstream cognitive tasks`. Fine-tuning refers to the process of taking a pre-trained language model and training it for a different but related task using specific data. The process is also known as transfer learning.

Large language models (LLMs) can seem to be foundation models that utilize deep learning in natural language processing (NLP) and natural language generation (NLG) tasks. Early examples of foundation models were pretrained language models like BERT, GPT-2, and XLNet. These models were trained on a large corpus of text data and then fine-tuned for specific tasks like question answering, text classification, and text summarization.

### **1.4. LLMs Application**

Large Language Models (LLMs) have a wide range of applications in natural language processing, including:

- Chatbots and virtual assistants
- Language translation services
- Text summarization
- Sentiment analysis
- Speech recognition
- Question answering
- Text completion and generation
- Spell checking and correction
- Named entity recognition
- Language modeling for speech synthesis

LLMs are also used in various industries such as healthcare, finance, and e-commerce for tasks such as customer service, fraud detection, and personalized recommendations.

### **1.5. Challenge and Limitations**

Large Language Models (LLMs) have been shown to perform well on a wide range of tasks. However, they are not without their limitations. Some of the main limitations and challenges of Large Language Models (LLMs) include:

- Ethical and societal implications, such as the potential for bias and misuse of the technology. It can lead to the potential to generate inappropriate, offensive, or toxic content.

- Maybe generating nonsensical, incoherent, or toxic responses - Without a full understanding of language and context, LLMs can sometimes produce responses that don't make logical sense or promote harmful ideas.

- The high computational resources required for training and inference and The need for large amounts of high-quality training data.

- The difficulty in incorporating external knowledge and context into the model. The challenge of dealing with rare or out-of-vocabulary words (like in some specialized domains such as medicine, laws, or in some languages)

## **2. General Architecture of LLMs**

Most of the early LLMs were created using RNN models with LSTMs and GRUs. However, they faced challenges, mainly in performing NLP tasks at massive scales. But, this is precisely where LLMs were expected to perform. This led to the creation of Transformers - the most popular architecture for LLMs.

### **2.1. Earlier Architecture of LLMs (RNN based)**

When it started, LLMs were largely created using self-supervised learning algorithms. Self-supervised learning refers to the `processing of unlabeled data to obtain useful representations` that can help with downstream learning tasks.

Quite often, self-supervised learning algorithms use a model based on an artificial neural network (ANN). We can create ANN using several architectures, but the most widely used architecture for LLMs were the recurrent neural network (RNN):

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Neural-Network-Architecture-RNN-1.jpg" >
  <br>
  <i>Recurrent Neural Network</i>
</p>

Now, RNNs can use their internal state to process variable-length sequences of inputs. An RNN `has both long-term memory and short-term memory`. There are variants of RNN like Long-short Term Memory (LSTM) and Gated Recurrent Units (GRU).

The LSTM architecture `helps an RNN when to remember and when to forget` important information. The GRU architecture is less complex, r`equires less memory to train, and executes faster than LSTM`. But GRU is generally more suitable for smaller datasets.

### **2.2. Problems with LSTMs & GRUs**

As we’ve seen earlier, LSTMs were introduced to bring memory into RNN. But an `RNN that uses LSTM units is very slow to train`. Moreover, we need to feed the data sequentially or serially for such architectures. This does not allow us to parallelize and use available processor cores.

Alternatively, an `RNN model with GRU trains faster but performs poorly on larger datasets`. Nevertheless, for a long time, LSTMs and GRUs remained the preferred choice for building complex NLP systems. However, such models also suffer from the vanishing gradient problem:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Vanishing-Gradient-Problem.jpg" >
  <br>
  <i>Vanishing Gradient Problem</i>
</p>

The `vanishing gradient problem is encountered in ANN` using gradient-based learning methods with backpropagation. In such methods, during each iteration of training, the weights receive an update proportional to the partial derivative of the error function concerning the current weight.

In some cases, like recurrent networks, the gradient becomes vanishingly small. This effectively prevents the weights from changing their value. This may even `prevent the neural network from training further`. These issues make the training of RNNs for NLP tasks practically inefficient.

### **2.3. Attention Mechanism**

Some of the problems with RNNs were partly addressed by adding the attention mechanism to their architecture. In recurrent architectures like LSTM, the amount of information that can be propagated is limited, and the window of retained information is shorter.

However, with the attention mechanism, this information window can be significantly increased. Attention is `a technique to enhance some parts of the input data while diminishing other parts`. The motivation behind this is that the network should devote more focus to the important parts of the data:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Self-Attention-Mechanism.jpg" >
  <br>
  <i>Attention Mechanism</i>
</p>

There is a `subtle difference between attention and self-attention`, but their motivation remains the same. While the attention mechanism refers to the ability to attend to different parts of another sequence, self-attention refers to the ability to attend to different parts of the current sequence.

Self-attention allows the model to access information from any input sequence element. In NLP applications, this `provides relevant information about far-away tokens`. Hence, the model can capture dependencies across the entire sequence without requiring fixed or sliding windows.

### **2.4. Word Embedding**

In NLP applications, how we represent the words or tokens appearing in a natural language is important. In LLM models, the input text is parsed into tokens, and each token is converted using a word embedding into a real-valued vector.

Word embedding is capable of capturing the meaning of the word in such a way that words that are closer in the vector space are expected to be similar in meaning. Further advances in word embedding also allow them to capture multiple meanings per word in different vectors:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Word-Embedding.jpg" >
  <br>
  <i>Word Embedding Example</i>
</p>

Word embeddings come in different styles, one of which is where the words are expressed as vectors of linguistic contexts in which the word occurs. Further, there are several approaches for generating word embeddings, of which the most popular one relies on neural network architecture.

In 2013, a team at Google published word2vec, a word embedding toolkit that uses a neural network model to learn word associations from a large corpus of text. Word and phrase embeddings have been shown to boost the performance of NLP tasks like syntactic parsing and sentiment analysis

### **2.5. Transformer Model**

The RNN models with attention mechanisms saw significant improvement in their performance. However, recurrent models are, by their nature, difficult to scale. But, the self-attention mechanism soon proved to be quite powerful, so much so that it did not even require recurrent sequential processing!

The introduction of transformers by the Google Brain team in 2017 is perhaps one of the most important inflection points in the history of LLMs. A transformer is a deep learning model that adopts the self-attention mechanism and processes the entire input all at once:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Neural-Network-Architecture-Transformer-Attention.jpg" >
  <br>
  <i>Transformer Model</i>
</p>

As a significant change to the earlier RNN-based models, transformers do not have a recurrent structure. With sufficient training data, the attention mechanism in the transformer architecture alone can match the performance of an RNN model with attention.

Another significant advantage of using the transformer model is that they are more parallelizable and require significantly less training time. This is exactly the sweet spot we require to build LLMs on a large corpus of text-based data with available resources.

We will discuss more about the transformer model in the other related articles.

### **2.6. Encoder-decoder Architecture**

Many ANN-based models for natural language processing are built using encoder-decoder architecture. For instance, seq2seq is a family of algorithms originally developed by Google. It turns one sequence into another sequence by using RNN with LSTM or GRU.

The original transformer model also used the encoder-decoder architecture. The encoder consists of encoding layers that process the input iteratively, one layer after another. The decoder consists of decoding layers that do the same thing to the encoder’s output:

<p align="center">
  <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2023/05/Neural-Network-Architecture-Transformer-Encoder-Decoder.jpg" >
  <br>
  <i>Encoder-Decoder Architecture in Transformer model</i>
</p>

The function of each encoder layer is to generate encodings that contain information about which parts of the input are relevant to each other. The output encodings are then passed to the next encoder as its input. Each encoder consists of a self-attention mechanism and a feed-forward neural network.

Further, each decoder layer takes all the encodings and uses their incorporated contextual information to generate an output sequence. Like encoders, each decoder consists of a self-attention mechanism, an attention mechanism over the encodings, and a feed-forward neural network.

## **3. Application Techniques for LLMs**

### **3.1. Prompting and prompt engineering**

`Promt` is the extual context that we feed into the model to help guide its generation of responses. Common types of prompts include conversational cues like "Let's discuss...", descriptive contexts, questions, continuations of previous dialog, etc. Prompts can be as brief as a few words or multiple paragraphs.

The output text of the model is known as the `completion`. The act of generating text above call `inference`. The full amount of text or the memory that is available to use for the prompt is called the `context window`.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214511691_f5b8076be5_o.png" >
  <br>
  <i>Promt</i>
</p>

Maybe sometime model performs well at the first try, we'll frequently encounter situations where the model doesn't produce the outcome that we want on the first try. we may have to revise the language in our prompt or the way that it's written several times to get the model to behave in the way that we want. This work to develop and improve the prompt is known as `prompt engineering`.

`Prompt engineering` refers to the process of carefully crafting textual prompts to elicit targeted behaviors from large language models (LLMs). The goal of prompt engineering is developing textual contexts that maximize the benefits of language models while mitigating risks like toxic, factually incorrect, or unaligned responses. It's an important technique for applying LLMs safely and ensuring user experiences meet expectations. Proper prompt design upfront helps avoid potential downstream issues.

But one powerful strategy to get the model to produce better outcomes is to include examples of the task that we want the model to carry out inside the prompt. Providing examples inside the context window is called `in-context learning`. There are the following types of in-context learning:

- `Zero-shot inference`: The model is given a prompt that describes the task, but no examples of the task. The model is expected to perform the task without any examples.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214901024_ba4fe7f7c2_o.png" >
  <br>
  <i>Zero-shot inference</i>
</p>

- `One-shot inference`: For smaller models, zero-shot inference maybe don't return a good result. So we have one-shot inference where the model is given a prompt that describes the task and one complete example of the task. The model is expected to perform the task with one example.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214511696_4eb3e8eb2c_o.png" >
  <br>
  <i>One-shot inference</i>
</p>

- `Few-shot inference`: For even smaller models that fail to perform the task with one example, we have few-shot inference. The model is given a prompt that describes the task and a few examples of the task.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214901049_a24286dba0_b.jpg" >
  <br>
  <i>Summary of in-context learning</i>
</p>

A drawback of in-context learning is that out context window have a limit amount of token so that we can't include too many examples. And also, in-context learning may not always work well for small models.

Generally, if we find that our model isn't performing well when including five or six examples, we should try fine-tuning our model instead. Fine-tuning performs additional training on the model using new data to make it more capable of the task we want it to perform.

### **3.2. Generative configuration parameters for inference**

In LLMs model, we have some parameters that we can use to control the model's behavior. These parameters are called `generative configuration parameters`. Each model exposes a set of configuration parameters that can influence the model's output during inference. Note that these are different than the training parameters which are learned during training time. Instead, these configuration parameters are invoked at inference time and give we control over things like the maximum number of tokens in the completion, and how creative the output is.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215063660_90c85c7e80_o.png" >
  <br>
  <i>Configuration parameters</i>
</p>

`Max new tokens` is the maximum number of tokens that the model can generate in the completion. This is useful for limiting the length of the completion. Note that max new tokens is not a hard limit on the number of tokens in the completion. The model may generate fewer tokens than the max new tokens value if it reaches a stopping condition.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53213670217_e29315c14b_o.png" >
  <br>
  <i>Max new tokens</i>
</p>

The output from the transformer's softmax layer is a probability distribution across the entire dictionary of words that the model uses. Most large language models by default will operate with `greedy decoding`. This is the simplest form of next-word prediction, where the model will always choose the word with the highest probability. This method can work very well for short generation but is susceptible to repeated words or repeated sequences of words.

If we want to generate text that's more natural, more creative and avoids repeating words, we need to use some other controls. `Random sampling` is the easiest way to introduce some variability. Instead of selecting the most probable word every time with random sampling, the model chooses an output word at random using the probability distribution to weight the selection. For example, in the illustration below, the word banana has a probability score of 0.02. With random sampling, this equates to a 2% chance that this word will be selected. By using this sampling technique, we reduce the likelihood that words will be repeated. However, depending on the setting, there is a possibility that the output may be too creative, producing words that cause the generation to wander off into topics or words that just don't make sense. Note that in some implementations, we may need to disable greedy and enable random sampling explicitly.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53213684897_2ce6f82b9a_o.png" >
  <br>
  <i>Greedy vs Random sampling</i>
</p>

Here we have `Top k`, `Top p` and `Temperature`, these are three parameters that can be used to control the model's creativity.

`Top k` focuses generation on a small subset of most likely tokens. At each step, the model considers only the k most likely next tokens according to the predicted probabilities (the top k highest probability tokens). The model then selects from these k options using the probability weighting. This method can help the model have some randomness while preventing the selection of highly improbable completion words. This in turn makes our text generation more likely to sound reasonable and to make sense.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53213684887_f04280c5b0_o.png" >
  <br>
  <i>Top k</i>
</p>

Alternatively, we can use the `Top p` setting to limit the random sampling to the predictions whose combined probabilities do not exceed p. For example, if we set p to equal 0.3, We add up probabilities from most to least likely until we cross the 0.3 threshold. The options are cake and donut since their probabilities of 0.2 and 0.1 add up to 0.3. The model then uses the random probability weighting method to choose from these tokens.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215079035_2a65f1ebd9_o.png" >
  <br>
  <i>Top p</i>
</p>

One more parameter that we can use to control the randomness of the model output is known as `temperature`. This parameter influences the shape of the probability distribution that the model calculates for the next token. The higher the temperature, the higher the randomness, and the lower the temperature, the lower the randomness. The temperature value is a scaling factor that's applied within the final softmax layer of the model that impacts the shape of the probability distribution of the next token. In contrast to the top k and top p parameters, changing the temperature actually alters the predictions that the model will make.

- If we choose a low value of temperature (less than one), the resulting probability distribution from the softmax layer is more strongly peaked with the probability being concentrated in a smaller number of words. Most of the probability here is concentrated on the word cake. The model will select from this distribution using random sampling and the resulting text will be less random and will more closely follow the most likely word sequences that the model learned during training.
- If instead we set the temperature to a higher value (greater than one), then the model will calculate a broader flatter probability distribution for the next token. Notice that in contrast to the blue bars, the probability is more evenly spread across the tokens. This leads the model to generate text with a higher degree of randomness and more variability in the output compared to a cool temperature setting. This can help we generate text that sounds more creative.
- If we leave the temperature value equal to one, this will leave the softmax function as default and the unaltered probability distribution will be used.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215079040_ab0294e98b_b.jpg" >
  <br>
  <i>Temperature</i>
</p>

### **3.3. Generative AI project lifecycle**

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214618491_b84df1562c_o.png" >
  <br>
  <i>Generative AI project lifecycle</i>
</p>

The basic Project lifecycle of a Generative AI deals with 4 core principles

- `Define Scope`: The most important step in any project is to define the scope as accurately and narrowly as we can. LLMs are capable of carrying out many tasks, but their abilities depend strongly on the size and architecture of the model. We should care about what function the LLM will have in a specific application.

- `Select model`: Next, we have to decide whether to train our own model from scratch or work with an existing base model. In general, we should start with an existing model, although there are some cases where we may have to train a model from scratch.

- `Adapt and align model`: The next step is to assess model's performance and carry out additional training if needed:
  - Prompt engineering can sometimes be enough to get well perform, we should likely start by trying in-context learning, using suitable examples.
  - If the model does not perform as well as we need, even with one or a few short inference, and in that case, we can try fine-tuning our model.
  - As models become more capable, it's important to ensure that they behave well and in a way that is aligned with human preferences in deployment. Here, we will perform an additional fine-tuning technique called reinforcement learning with human feedback, which can help to make sure that model behaves well.
  - An important aspect of all of these techniques is evaluation. We should use some metrics and benchmarks to determine how well our model is performing or how well aligned it is to our preferences.

- `Application integration`: Finally, when we've got a model that is good perform and well aligned, we can deploy it into an infrastructure and integrate an application. At this stage, an important step is to optimize our model for deployment to ensure of efficient compute resources and good experience for the users.

### **3.4. Pre-training large language models**

#### **3.4.1. Pre-training LLMs at a high level**

The initial training process for LLMs is often referred to as pre-training. LLMs encode a deep statistical representation of language. This understanding is developed during the models pre-training phase when the model learns from vast amounts of unstructured textual data. This can be gigabytes, terabytes, and even petabytes of text. This data is pulled from many sources, including scrapes off the Internet and corpora of texts that have been assembled specifically for training language models.

In this self-supervised learning step, the model internalizes the patterns and structures present in the language. These patterns then enable the model to complete its training objective, which depends on the architecture of the model. During pre-training, the model weights get updated to minimize the loss of the training objective. The encoder generates an embedding or vector representation for each token. Pre-training also requires a large amount of compute and the use of GPUs.

Note, when we scrape training data from public sites such as the Internet, we often need to process the data to increase quality, address bias, and remove other harmful content. As a result of this data quality curation, often only 1-3% of tokens are used for pre-training.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214984308_c255a486d8_b.jpg" >
  <br>
  <i>LLM pre-training at a high level</i>
</p>

#### **3.4.2. Transformer model types**

There were three variance of the transformer model: encoder-only, encoder-decoder models, and decode-only. Each of these is trained on a different objective, and so learns how to carry out different tasks.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214984298_f977d52c0c_o.png" >
  <br>
  <i>Transformer model types</i>
</p>

`Encoder-only models` are also known as `Autoencoding models`, and they are pre-trained using masked language modeling. Here, tokens in the input sequence or randomly mask, and the training objective is to predict the mask tokens in order to reconstruct the original sentence. This is also called a denoising objective. Autoencoding models spilled bi-directional representations of the input sequence, meaning that the model has an understanding of the full context of a token and not just of the words that come before. Encoder-only models are ideally suited to task that benefit from this bi-directional contexts. we can use them to carry out sentence classification tasks, for example, sentiment analysis or token-level tasks like named entity recognition or word classification. Some well-known examples of an autoencoder model are BERT and RoBERTa.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214984358_f0283ba142_b.jpg" >
  <br>
  <i>Autoencoding models</i>
</p>

`Decoder-only` or `autoregressive models`, which are pre-trained using causal language modeling. Here, the training objective is to predict the next token based on the previous sequence of tokens. Decoder-based autoregressive models, mask the input sequence and can only see the input tokens leading up to the token in question. The model has no knowledge of the end of the sentence. The model then iterates over the input sequence one by one to predict the following token. In contrast to the encoder architecture, this means that the context is unidirectional. By learning to predict the next token from a vast number of examples, the model builds up a statistical representation of language. Models of this type make use of the decoder component off the original architecture without the encoder. Decoder-only models are often used for text generation, although larger decoder-only models show strong zero-shot inference abilities, and can often perform a range of tasks well. Well known examples of decoder-based autoregressive models are GBT and BLOOM.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215066264_2bf7b13ca8_o.png" >
  <br>
  <i>Autoregressive models</i>
</p>

The final variation of the transformer model is the sequence-to-sequence model that uses both the encoder and decoder parts off the original transformer architecture. The exact details of the pre-training objective vary from model to model. A popular sequence-to-sequence model T5, pre-trains the encoder using span corruption, which masks random sequences of input tokens. Those mass sequences are then replaced with a unique `Sentinel token`, shown here as x. `Sentinel tokens` are special tokens added to the vocabulary, but do not correspond to any actual word from the input text. The decoder is then tasked with reconstructing the mask token sequences auto-regressively. The output is the Sentinel token followed by the predicted tokens. we can use sequence-to-sequence models for translation, summarization, and question-answering. They are generally useful in cases where we have a body of texts as both input and output.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214673701_09dca5b6d6_b.jpg" >
  <br>
  <i>Sequence-to-sequence models</i>
</p>

To summarize, here's a quick comparison of the different model architectures and the targets off the pre-training objectives:

- Autoencoding models are pre-trained using masked language modeling. They correspond to the encoder part of the original transformer architecture, and are often used with sentence classification or token classification.
- Autoregressive models are pre-trained using causal language modeling. Models of this type make use of the decoder component of the original transformer architecture, and often used for text generation.
- Sequence-to-sequence models use both the encoder and decoder part off the original transformer architecture. The exact details of the pre-training objective vary from model to model. The T5 model is pre-trained using span corruption. Sequence-to-sequence models are often used for translation, summarization, and question-answering.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53214984338_2730a89474_o.png" >
  <br>
  <i>Model architectures and pre-training objectives</i>
</p>

## **4. Fine-tuning LLMs**

Earlier, we see the main drawback of in-context learning is that context window have a limit amount of token so that we can't include too many examples. And also, in-context learning may not always work well for small models even when we include five or six examples.

Luckily, we have another technique called `fine-tuning` to solve this problem. Fine-tuning performs additional training on the model using new data to make it more capable of the task we want it to perform.

### **4.1. Fine-tuning LLMs at a high level**

In contrast to pre-training, where we train the LLM using vast amounts of unstructured textual data via selfsupervised learning, fine-tuning is a supervised learning process where we use a data set of labeled examples to update the weights of the LLM. The labeled examples are prompt - completion pairs, the fine-tuning process extends the training of the model to improve its ability to generate good completions for a specific task.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215573767_45935edce0_o.png" >
  <br>
  <i>LLM fine-tuning at a high level</i>
</p>

### **4.2. Instruction Tuning**

#### **4.2.1. What is Instruction Tuning?**

One strategy, known as instruction fine tuning, is particularly good at improving a model's performance on a variety of tasks. Instruction fine-tuning trains the model using examples that demonstrate how it should respond to a specific instruction.

The instruction in both examples is classify this review, and the desired completion is a text string that starts with sentiment followed by either positive or negative. The data set we use for training includes many pairs of prompt completion examples for the task we're interested in, each of which includes an instruction.

For example, if we want to fine tune our model to improve its summarization ability, we'd build up a data set of examples that begin with the instruction summarize, the following text or a similar phrase. And if we are improving the model's translation skills, our examples would include instructions like translate this sentence. These prompt completion examples allow the model to learn to generate responses that follow the given instructions.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216840104_7c02740887_o.png" >
  <br>
  <i>Instruction fine-tuning</i>
</p>

Instruction fine-tuning, where all of the model's weights are updated is known as full fine-tuning. The process results in a new version of the model with updated weights. It is important to note that just like pre-training, full fine tuning requires enough memory and compute budget to store and process all the gradients, optimizers and other components that are being updated during training.

#### **4.2.2. Prepare training data**

The first step is to prepare our training data. There are many publicly available datasets that have been used to train earlier generations of language models, although most of them are not formatted as instructions. Luckily, developers have assembled prompt template libraries that can be used to take existing datasets, for example, the large data set of Amazon product reviews and turn them into instruction prompt datasets for fine-tuning. Prompt template libraries include many templates for different tasks and different data sets. Here are three prompts that are designed to work with the Amazon reviews dataset and that can be used to fine tune models for classification, text generation and text summarization tasks.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216451661_b978510587_o.png" >
  <br>
  <i>Sample prompt instruction templates</i>
</p>

we can see that in each case we pass the original review, here called review_body, to the template, where it gets inserted into the text that starts with an instruction like predict the associated rating, generate a star review, or give a short sentence describing the following product review. The result is a prompt that now contains both an instruction and the example from the data set.

Once we have our instruction data set ready, as with standard supervised learning, we divide the data set into training validation and test splits.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216840099_ea58bbf788_o.png" >
  <br>
  <i>Split prepared instruction data into train/val/test</i>
</p>

#### **4.2.3. Instruction tuning process**

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216451651_688d9dcc8a_o.png" >
  <br>
  <i>Overall process of Instruction tuning LLMs</i>
</p>

During fine tuning, we select prompts from our training data set and pass them to the LLM, which then generates completions. Next, we compare the LLM completion with the response specified in the training data.

we can see here that the model didn't do a great job, it classified the review as neutral, which is a bit of an understatement. The review is clearly very positive. Remember that the output of an LLM is a probability distribution across tokens. So we can compare the distribution of the completion and that of the training label and use the standard crossentropy function to calculate loss between the two token distributions. And then use the calculated loss to update our model weights in standard backpropagation. we'll do this for many batches of prompt completion pairs and over several epochs, update the weights so that the model's performance on the task improves.

As in standard supervised learning, we can define separate evaluation steps to measure our LLM performance using the holdout validation data set. This will give we the validation accuracy, and after we've completed our fine tuning, we can perform a final performance evaluation using the holdout test data set. This will give we the test accuracy. The

fine-tuning process results in a new version of the base model, often called an instruct model that is better at the tasks we are interested in. Fine-tuning with instruction prompts is the most common way to fine-tune LLMs these days. From this point on, when we hear or see the term fine-tuning, we can assume that it always means instruction fine tuning.

#### **4.2.4. Instruction fine-tuning for a single task**

While LLMs can perform many different language tasks within a single model, our application may only need to perform a single task. In this case, we can fine-tune a pre-trained model to improve performance on only the task that is of interest to we.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217005155_4fb40f6475_o.png" >
  <br>
  <i>Instruction fine-tuning for a single task</i>
</p>

For example, summarization using a dataset of examples for that task. Interestingly, good results can be achieved with relatively few examples. Often just 500-1,000 examples can result in good performance in contrast to the billions of pieces of texts that the model saw during pre-training.

However, there is a potential downside to fine-tuning on a single task. The process may lead to a phenomenon called `catastrophic forgetting`. Catastrophic forgetting leads to great performance on the single fine-tuning task, it can degrade performance on other tasks. This happens because the full fine-tuning process modifies the weights of the original LLM.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216887474_e4db33a37e_o.png" >
  <img src="https://live.staticflickr.com/65535/53215621407_717a764aaf_o.png" >
  <br>
  <i>Catastrophic forgetting example</i>
</p>

For the above example, while fine-tuning can improve the ability of a model to perform sentiment analysis on a review and result in a quality completion, the model may forget how to do other tasks.

How to avoid catastrophic forgetting? First of all, it's important to decide whether catastrophic forgetting actually impacts our use case.

- If all we need is reliable performance on the single task we fine-tuned on, it may not be an issue that the model can't generalize to other tasks.
- If we do want or need the model to maintain its multitask generalized capabilities, we can perform fine-tuning on multiple tasks at one time. Good multitask fine-tuning may require 50-100,000 examples across many tasks, and so will require more data and compute to train.
- Our second option is to perform parameter efficient fine-tuning (PEFT) instead of full fine-tuning. PEFT shows greater robustness to catastrophic forgetting since this technique preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters. We will discuss about PEFT in the later part.

#### **4.2.5. Instruction fine-tuning for multiple tasks**

Multitask fine-tuning is an extension of single task fine-tuning, where the training dataset is comprised of example inputs and outputs for multiple tasks.

Here, the dataset contains examples that instruct the model to carry out a variety of tasks, including summarization, review rating, code translation, and entity recognition. we train the model on this mixed dataset so that it can improve the performance of the model on all the tasks simultaneously, thus avoiding the issue of catastrophic forgetting. Over many epochs of training, the calculated losses across examples are used to update the weights of the model, resulting in an instruction tuned model that is learned how to be good at many different tasks simultaneously.

One drawback to multitask fine-tuning is that it requires a lot of data. we may need as many as 50-100,000 examples in our training set. However, it can be really worthwhile and worth the effort to assemble this data. The resulting models are often very capable and suitable for use in situations where good performance at many tasks is desirable.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216513706_98a01f27af_o.png" >
  <br>
  <i>Instruction fine-tuning for a multiple tasks</i>
</p>

Instruct model variance differ based on the datasets and tasks used during fine-tuning. One example is the `FLAN family of models`. FLAN, which stands for fine-tuned language net, is a specific set of instructions used to fine-tune different models. One example of a prompt dataset used for summarization tasks is SAMSum which is a dataset with 16,000 messenger like conversations with summaries.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216902639_e0b7d74f26_o.png" >
  <br>
  <i>SAMSum prompt template</i>
</p>

### **4.2. Parameter efficient fine-tuning (PEFT)**

#### **4.2.1. PEFT vs Full Fine-tuning**

Training LLMs is computationally intensive. Full fine-tuning requires memory not just to store the model, but various other parameters that are required during the training process. Even if our computer can hold the model weights, which are now on the order of hundreds of gigabytes for the largest models, we must also be able to allocate memory for optimizer states, gradients, forward activations, and temporary memory throughout the training process. These additional components can be many times larger than the model and can quickly become too large to handle on consumer hardware.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216863203_af42ce91ec_o.png" >
  <br>
  <i>Full fine-tuning problems</i>
</p>

In contrast to full fine-tuning where every model weight is updated during supervised learning, PEFT methods only update a small subset of parameters. Some path techniques freeze most of the model weights and focus on fine tuning a subset of existing model parameters, for example, particular layers or components. Other techniques don't touch the original model weights at all, and instead add a small number of new parameters or layers and fine-tune only the new components.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216551471_e05a85364a_o.png" >
  <img src="https://live.staticflickr.com/65535/53215675922_52dd0ab86f_o.png" >
  <br>
  <i>PEFT methods is more efficient than full fine-tuning</i>
</p>

With PEFT, most or all of the LLM weights are kept frozen. As a result, the number of trained parameters is much smaller than the number of parameters in the original LLM. In some cases, just 15-20% of the original LLM weights. This makes the memory requirements for training much more manageable. And because the original LLM is only slightly modified or left unchanged, PEFT is less prone to the catastrophic forgetting problems of full fine-tuning.

Full fine-tuning results in a new version of the model for every task we train on. Each of these is the same size as the original model, so it can create an expensive storage problem if we're fine-tuning for multiple tasks. With PEFT, we train only a small number of weights, which results in a much smaller footprint overall, as small as megabytes depending on the task. The new parameters are combined with the original LLM weights for inference. The PEFT weights are trained for each task and can be easily swapped out for inference, allowing efficient adaptation of the original model to multiple tasks.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216940849_49a458fdb6_o.png" >
  <img src="https://live.staticflickr.com/65535/53216863208_6d72bc0dc9_o.png" >
  <br>
  <i>PEFT saves space and is more flexible than full fine-tuning</i>
</p>

There are several methods we can use for parameter efficient fine-tuning, each with trade-offs on parameter efficiency, memory efficiency, training speed, model quality, and inference costs. There are three main classes of PEFT methods.

- `Selective methods` are those that fine-tune only a subset of the original LLM parameters. There are several approaches that we can take to identify which parameters we want to update. we have the option to train only certain components of the model or specific layers, or even individual parameter types. Researchers have found that the performance of these methods is mixed and there are significant trade-offs between parameter efficiency and compute efficiency so we will not discuss about it in this report.

- `Reparameterization methods` also work with the original LLM parameters, but reduce the number of parameters to train by creating new low rank transformations of the original network weights. A commonly used technique of this type is LoRA.

- `Additive methods` carry out fine-tuning by keeping all of the original LLM weights frozen and introducing new trainable components. Here there are two main approaches.
  - `Adapter methods` add new trainable layers to the architecture of the model, typically inside the encoder or decoder components after the attention or feed-forward layers.
  - `Soft prompt methods`, on the other hand, keep the model architecture fixed and frozen, and focus on manipulating the input to achieve better performance. This can be done by adding trainable parameters to the prompt embeddings or keeping the input fixed and retraining the embedding weights. In this report, we'll take a look at a specific soft prompts technique called `prompt tuning`.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216908848_4e35e9e5c7_o.png" >
  <br>
  <i>PEFT methods</i>
</p>

#### **4.2.1. PEFT techniques 1: LORA**

`Low-rank Adaptation`, or `LoRA` for short, is a parameter-efficient fine-tuning technique that falls into the reparameterization category.

LoRA is a strategy that reduces the number of parameters to be trained during fine-tuning by freezing all of the original model parameters and then injecting a pair of rank decomposition matrices alongside the original weights. The dimensions of the smaller matrices are set so that their product is a matrix with the same dimensions as the weights they're modifying. we then keep the original weights of the LLM frozen and train the smaller matrices using the same supervised learning process.

For inference, the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights. Then, we add this to the original weights and replace them in the model with these updated values. we now have a LoRA fine-tuned model that can carry out our specific task. Because this model has the same number of parameters as the original, there is little to no impact on inference latency.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216598686_4a2962010e_o.png" >
  <br>
  <i>LoRA Process</i>
</p>

Researchers have found that applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve performance gains. However, in principle, we can also use LoRA on other components like the feed-forward layers. But since most of the parameters of LLMs are in the attention layers, we get the biggest savings in trainable parameters by applying LoRA to these weights matrices.

For an example that used in the transformer architecture described in the Attention is All we Need paper. By updating the weights of these new low-rank matrices instead of the original weights, we can reduce 86% parameters in training.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216912613_57662f24a4_o.png" >
  <br>
  <i>LoRA Efficient example</i>
</p>

Since the rank-decomposition matrices are small, we can fine-tune a different set for each task and then switch them out at inference time by updating the weights. Suppose we train a pair of LoRA matrices for a specific task A. To carry out inference on this task, we would multiply these matrices together and then add the resulting matrix to the original frozen weights. we then take this new summed weights matrix and replace the original weights where they appear in our model. we can then use this model to carry out inference on Task A. If instead, we want to carry out a different task, say Task B, we simply take the LoRA matrices we trained for this task, calculate their product, and then add this matrix to the original weights and update the model again.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217105405_e9c0df6177_o.png" >
  <br>
  <i>LoRA Adaptable for multiple tasks</i>
</p>

Reseacher has clarify that LoRA fine-tuning is not always as effective as full fine-tuning. In some cases, it can lead to a loss in performance. However, it is a good trade off when we need to fine-tune a model for a task and we have limited compute resources. It can also be a good option when we want to fine-tune a model for multiple tasks and we want to be able to switch between tasks quickly.

Choosing LoRA rank is a hyperparameter that we can tune. In general, the higher the rank, the more parameters we have to train, and the more compute we need. However, higher rank matrices can capture more information from the original weights and so can lead to better performance. But there is a point that the performance gain is not worth the additional compute cost. So we should experiment with different rank values to find the best trade-off between performance and compute efficiency.

#### **4.2.2. PEFT techniques 2: Soft Prompt Tuning**

Prompt tuning sounds a bit like prompt engineering, but they are quite different from each other. With prompt engineering, we work on the language of our prompt to get the completion we want. This could be as simple as trying different words or phrases or more complex, like including examples for one or Few-shot Inference. The goal is to help the model understand the nature of the task we're asking it to carry out and to generate a better completion. With prompt tuning, we add additional trainable tokens to our prompt and leave it up to the supervised learning process to determine their optimal values.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217013089_278acb5dbd_o.png" >
  <br>
  <i>Prompt tuning adds trainable “soft prompt” to inputs</i>
</p>

The set of trainable tokens is called a soft prompt, and it gets prepended to embedding vectors that represent our input text. The soft prompt vectors have the same length as the embedding vectors of the language tokens. And including somewhere between 20 and 100 virtual tokens can be sufficient for good performance. The tokens that represent natural language are hard in the sense that they each correspond to a fixed location in the embedding vector space. However, the soft prompts are not fixed discrete words of natural language. Instead, we can think of them as virtual tokens that can take on any value within the continuous multidimensional embedding space. And through supervised learning, the model learns the values for these virtual tokens that maximize performance for a given task.

In full fine tuning, the training data set consists of input prompts and output completions or labels. The weights of the large language model are updated during supervised learning. In contrast with prompt tuning, the weights of the large language model are frozen and the underlying model does not get updated. Instead, the embedding vectors of the soft prompt gets updated over time to optimize the model's completion of the prompt. Prompt tuning is a very parameter efficient strategy because only a few parameters are being trained. In contrast with the millions to billions of parameters in full fine tuning.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216933748_0cf6db987c_o.png" >
  <br>
  <i>Full Fine-tuning vs prompt tuning</i>
</p>

we can train a different set of soft prompts for each task and then easily swap them out at inference time. we can train a set of soft prompts for one task and a different set for another. To use them for inference, we prepend our input prompt with the learned tokens to switch to another task, we simply change the soft prompt. Soft prompts are very small on disk, so this kind of fine tuning is extremely efficient and flexible. we'll notice the same LLM is used for all tasks, all we have to do is switch out the soft prompts at inference time.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216623831_3bb980dd6a_o.png" >
  <br>
  <i>Soft prompt tuning for multiple task</i>
</p>

## **5. Improve performance and alignment of LLMs using Reinforcement Learning From Human Feedback (RLHF)**

### **5.2. Aligning LLMs with human preferences**

The goal of fine-tuning with instructions, including path methods, is to further train our models so that they better understand human like prompts and generate more human-like responses. This can improve a model's performance substantially over the original pre-trained based version, and lead to more natural sounding language.

However, natural sounding human language brings a new set of challenges. There are lot of information about LLMs behaving badly. Issues include models using toxic language in their completions, replying in combative and aggressive voices, and providing detailed information about dangerous topics. These problems exist because large models are trained on vast amounts of texts data from the Internet where such language appears frequently.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215878552_f98048b7ce_o.png" >
  <br>
  <i>Examples of LLMs behaving badly</i>
</p>

The above human preferences on LLMs output: helpfulness, honesty, and harmlessness are sometimes collectively called `HHH`, and are a set of principles that guide developers in the responsible use of AI. Additional fine-tuning with human feedback helps to better align models with human preferences and to increase the helpfulness, honesty, and harmlessness of the completions. This further training can also help to decrease the toxicity, often models responses and reduce the generation of incorrect information.

### **5.3. How RLHF works to improve alignment of LLMs**

Reinforcement learning from human feedback is a technique to finetune LLMs with human feedback. Let's consider the task of text summarization, where we use the model to generate a short piece of text that captures the most important points in a longer article. In 2020, researchers at OpenAI published a paper that explored the use of fine-tuning with human feedback to train a model to write short summaries of text articles. The results show that a model fine-tuned on human feedback produced better responses than a pretrained model, an instruct fine-tuned model, and even the reference human baseline. A popular technique to finetune large language models with human feedback is called reinforcement learning from human feedback, or RLHF for short.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217314915_6036c8b267_o.png" >
  <br>
  <i>Result of fine-tuning with human feedback</i>
</p>

As the name suggests, RLHF uses reinforcement learning to finetune the LLM with human feedback data, resulting in a model that is better aligned with human preferences. By this way, RLHF can help minimize the potential for harm, maximize usefulness and relevance of LLMs.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217196424_9231d7d97b_o.png" >
  <br>
  <i>Objective of RLHF</i>
</p>

For reviewing, Reinforcement learning is a type of machine learning in which an agent learns to make decisions related to a specific goal by taking actions in an environment, with the objective of maximizing some notion of a cumulative reward. In this framework, the agent continually learns from its experiences by taking actions, observing the resulting changes in the environment, and receiving rewards or penalties, based on the outcomes of its actions. By iterating through this process, the agent gradually refines its strategy or policy to make better decisions and increase its chances of success. Initially, the agent takes a random action which leads to a new state. From this state, the agent proceeds to explore subsequent states through further actions. The series of actions and corresponding states form a playout, often called a rollout. As the agent accumulates experience, it gradually uncovers actions that yield the highest long-term rewards.

---

*Now, how the reinforcement learning can be extended to the case of fine-tuning large language models with RLHF?*

In this case, the agent's policy that guides the actions is the LLM, and its objective is to generate text that is perceived as being aligned with the human preferences, for example, helpful, accurate, and non-toxic.

The environment is the context window of the model where text can be entered via a prompt. The state that the model considers before taking an action is the current context. That means any text currently contained in the context window.

The action here is the act of generating text. This could be a single word, a sentence, or a longer form text, depending on the task specified by the user. The action space is the token vocabulary, meaning all the possible tokens that the model can choose from to generate the completion. How an LLM decides to generate the next token in a sequence, depends on the statistical representation of language that it learned during its training. At any given moment, the action that the model will take, meaning which token it will choose next, depends on the prompt text in the context and the probability distribution over the vocabulary space.

The reward is assigned based on how closely the completions align with human preferences. Given the variation in human responses to language, determining the reward is more complicated. One way to do this is to have a human evaluate all of the completions of the model against some alignment metric, such as determining whether the generated text is toxic or non-toxic. This feedback can be represented as a scalar value, either a zero or a one.

The LLM weights are then updated iteratively to maximize the reward obtained from the human classifier, enabling the model to generate non-toxic completions.

However, obtaining human feedback can be time consuming and expensive. As a practical and scalable alternative, we can use an additional model, known as the `reward model`, to classify the outputs of the LLM and evaluate the degree of alignment with human preferences. We'll start with a smaller number of human examples to train the secondary model by traditional supervised learning methods. Once trained, we'll use the reward model to assess the output of the LLM and assign a reward value, which in turn gets used to update the weights off the LLM and train a new human aligned version. Exactly how the weights get updated as the model completions are assessed, depends on the algorithm used to optimize the policy.The reward model is the central component of the reinforcement learning process. It encodes all of the preferences that have been learned from human feedback, and it plays a central role in how the model updates its weights over many iterations.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215932087_1491c0e98a_b.jpg" >
  <br>
  <i>Flow of fine-tuning LLMs with RLHF</i>
</p>

### **5.4. Obtaining human feedback**

The first step in fine-tuning an LLM with RLHF is to select a model to work with and use it to prepare a data set for human feedback. The model we choose should have some capability to carry out the task like text summarization, question answering or something else. In general, it's easier to start with an instruct model that has already been fine tuned across many tasks and has some general capabilities. we'll then use this LLM along with a prompt dataset to generate a number of different responses for each prompt. The prompt dataset is comprised of multiple prompts, each of which gets processed by the LLM to produce a set of completions.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217136603_fa7cdcdcfd_o.png" >
  <br>
  <i>Prepare dataset for human feedback</i>
</p>

The next step is to collect feedback from human labelers on the completions generated by the LLM. First, we must decide what criterion we want the humans to assess the completions on. This could be any of the issues like helpfulness or toxicity. Once we've decided, we will then ask the labelers to assess each completion in the data set based on that criterion.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217217284_a476afdd15_o.png" >
  <br>
  <i>Example of collecting human feedback</i>
</p>

For the above example, we pass this prompt to the LLM, which then generates 3 different completions. The task for labelers is to rank the 3 completions in order of helpfulness from the most helpful to least helpful. So here the labeler will probably decide that completion two is the most helpful. This process then gets repeated for many prompt completion sets, building up a data set that can be used to train the reward model that will ultimately carry out this work instead of the humans. The same prompt completion sets are usually assigned to multiple human labelers to establish consensus and minimize the impact of poor labelers in the group. Like the third labeler here, whose responses disagree with the others and may indicate that they misunderstood the instructions, this is actually a very important point. The clarity of our instructions can make a big difference on the quality of the human feedback we obtain.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217217294_d7ca21ddbc_b.jpg" >
  <br>
  <i>Sample instructions for human labelers</i>
</p>

Once your human labelers have completed their assessments off the prompt completion sets, you have all the data you need to train the reward model. Which you will use instead of humans to classify model completions during the reinforcement learning finetuning process.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216828061_03576f4f7a_o.png" >
  <br>
  <i>Ranking data into a pairwise comparison</i>
</p>

Before you start to train the reward model, you need to convert the ranking data into a pairwise comparison of completions. In other words, all possible pairs of completions from the available choices to a prompt should be classified as 0 or 1 score. In above example, there are three completions to a prompt, and the ranking assigned by the human labelers was 2, 1, 3, where 1 is the highest rank corresponding to the most preferred response. With the three different completions, there are three possible pairs. Depending on the number N of alternative completions per prompt, you will have N choose two combinations. For each pair, you will assign a reward of 1 for the preferred response and a reward of 0 for the less preferred response. Then you'll reorder the prompts so that the preferred option comes first. This is an important step because the reward model expects the preferred completion, which is referred to as $`Y_j`$ first. Once you have completed this data, restructuring, the human responses will be in the correct format for training the reward model. Note that while like or dislike feedback is often easier to gather than ranking feedback, ranked feedback gives more prompt completion data to train your reward model. As you can see, here you get three prompt completion pairs from each.

### **5.5. Training the reward model**

Now, you have everything you need to train the reward model. By the time you're done training the reward model, you won't need to include any more humans in the loop. Instead, the reward model will effectively take the place of humans and automatically choose the preferred completion during the human feedback process. This reward model is usually also a language model. For example, a BERT is trained using supervised learning methods on the pairwise comparison data that you prepared from the human labeler's assessment of the prompts. For a given prompt X, the reward model learns to favor the human-preferred completion $`y_j`$, while minimizing the log of sigmoid of the reward difference $`r_j - r_k`$

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217237519_cb88078fd3_o.png" >
  <br>
  <i>Training the reward model</i>
</p>

Once the reward model has been trained on the human rank prompt-completion pairs, you can use it as a binary classifier to provide a set of logits across the positive and negative classes. Logits are the unnormalized model outputs before applying any activation function. If you apply a Softmax function to the logits, you will get the probabilities

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215973357_b8812817ec_o.png" >
  <br>
  <i>Use the reward model</i>
</p>

If you want to detoxify your LLMs, and the reward model needs to identify if the completion contains hate speech. In this case, the positive class want to optimize for is not hate and negative class you want to avoid is hate. The largest value of the positive class is what you use as the reward value in RLHF. The first example shows a good reward for non-toxic completion and the second example shows a bad reward for toxic completion.

### **5.6. Fine-tuning with RLHF**

Here, we start with a model that already has good performance on your task of interests. We'll work to align an instruction fine-tuned LLM.

- First, you'll pass a prompt from your prompt dataset. In this case, "a dog is", to the instruct LLM, which then generates a completion, in this case "a furry animal".

- Next, you sent this completion, and the original prompt to the reward model as the prompt-completion pair. The reward model evaluates the pair and returns a reward value.

- Then pass this reward value to the reinforcement learning algorithm to update the weights of the LLM.

These series of steps together forms a single iteration of the RLHF process. These iterations continue for a given number of epochs, similar to other types of fine tuning. You will continue this iterative process until your model is aligned based on some evaluation criteria. For example, reaching a threshold value for the helpfulness you defined. You can also define a maximum number of steps, for example, 20,000 as the stopping criteria.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217254279_f5e7dd5f10_o.png" >
  <img src="https://live.staticflickr.com/65535/53217173803_17b7c76cc4_o.png" >
  <img src="https://live.staticflickr.com/65535/53216864216_0082892685_o.png" >
  <img src="https://live.staticflickr.com/65535/53217173798_fb4367b2eb_o.png" >
  <br>
  <i>Fine-tuning with RLHF</i>
</p>

There are also several different algorithms that you can use for this part of the RLHF process. A popular choice is proximal policy optimization or PPO for short.

## **6. Evaluating LLMs**

### **6.1. ROUGE (Recall-Oriented Understudy for Gissing Evaluation)**

ROUGE is a set of metrics used for evaluating the quality of summaries. It compares the generated summary with one or more reference summaries and calculates precision, recall, and F1-score (Figure 4). ROUGE scores provide insights into the summary generation capabilities of the language model.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53216054612_8e9fb4cbd1_o.png" >
  <img src="https://live.staticflickr.com/65535/53216054617_79f398acba_o.png" >
  <img src="https://live.staticflickr.com/65535/53217237518_726d314ee3_o.png" >
  <br>
  <i>ROUGE examples</i>
</p>

### **6.2. BLEU (Bilingual Evaluation Understudy)**

BLEU is a metric commonly used in machine translation tasks. It compares the generated output with one or more reference translations and measures the similarity between them. BLEU scores range from 0 to 1, with higher scores indicating better performance.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53217317484_624f7f62f5_o.png" >
  <br>
  <i>BLEU example</i>
</p>

### **6.3. Perplexity**

The most commonly used measure of a language model's performance is its perplexity on a given text corpus. Perplexity is a measure of how well a model is able to predict the contents of a dataset; the higher the likelihood the model assigns to the dataset, the lower the perplexity. Mathematically, perplexity is defined as the exponential of the average negative log likelihood per token:

```math
log(Permplexity) = -\frac{1}{N}\sum_{i=1}^{N}log(P(token_i|context for token_i)) = -\frac{1}{N}\sum_{i=1}^{N}log(P(token_i|token_1, token_2, ..., token_{i-1}))
```

here N is the number of tokens in the text corpus, and "context for token i" depends on the specific type of LLM used. If the LLM is autoregressive, then "context for token i" is the segment of text appearing before token i. If the LLM is masked, then "context for token i" is the segment of text surrounding token i.

Because language models may overfit to their training data, models are usually evaluated by their perplexity on a test set of unseen data. This presents particular challenges for the evaluation of large language models. As they are trained on increasingly large corpora of text largely scraped from the web, it becomes increasingly likely that models' training data inadvertently includes portions of any given test set.

<p align="center">
  <img src="https://images.surferseo.art/42b4e02c-2bfb-4955-bd5a-f55bf90465fb.png" >
  <br>
  <i>Perplexity example</i>
</p>

### **6.4. Human evaluation**

The evaluation process includes enlisting human evaluators who assess the quality of the language model’s output. These evaluators rate 3 the generated responses based on different criteria, including:

- Relevance
- Fluency
- Coherence
- Overall quality.

This approach offers subjective feedback on the model’s performance. However, it can be time-consuming and expensive, especially for large-scale evaluations.

<p align="center">
  <img src="https://images.surferseo.art/dc475e7f-26df-46e0-b5a2-ee2cc7f7f906.png" >
  <br>
  <i>Example of The human evaluator uses both models simultaneously to decide which model is better</i>
</p>

### **6.4. Benchmarking**

Beside the above evaluation methods, benchmarking is also a popular method for evaluating LLMs. Benchmarking is the process of comparing the performance of a model against a set of standard tasks or datasets. Benchmarking provides a standardized way to compare the performance of different models. It also helps to identify the strengths and weaknesses of a model and to understand how it performs on different tasks. However, benchmarking has some limitations. For example, it can be difficult to design a benchmark that captures the full range of a model’s capabilities. Additionally, benchmarking may not be able to capture the nuances of a model’s performance on a specific task. Therefore, it is important to use benchmarking in conjunction with other evaluation methods.

| Framework Name | Factors Considered for Evaluation | Url Link|
| --- | --- | --- |
| Big Bench | Generalization abilities | <https://github.com/google/BIG-bench> |
| GLUE Benchmark | Grammar, Paraphrasing, Text Similarity, Inference, Textual Entailment, Resolving Pronoun References | <https://gluebenchmark.com/> |
| SuperGLUE Benchmark | Natural Language Understanding, Reasoning, Understanding complex sentences beyond training data, Coherent and Well-Formed Natural Language Generation, Dialogue with Human Beings, Common Sense Reasoning (Everyday Scenarios and Social Norms and Conventions), Information Retrieval, Reading Comprehension | <https://super.gluebenchmark.com/> |
| OpenAI Moderation API | Filter out harmful or unsafe content | <https://platform.openai.com/docs/api-reference/moderations> |
| MMLU | Language understanding across various tasks and domains | <https://github.com/hendrycks/test> |

### **6.5. Challenges with existing LLM evaluation methods**

While existing evaluation methods for Large Language Models (LLMs) provide valuable insights, they are not perfect. The common issues associated with them are:

- *Over-reliance on perplexity*: Perplexity measures how well a model predicts a given text but does not capture aspects such as coherence, relevance, or context understanding. Therefore, relying solely on perplexity may not provide a comprehensive assessment of an LLM’s quality.

- *Subjectivity in human evaluations*: Human evaluation is a valuable method for assessing LLM outputs, but it can be subjective and prone to bias. Different human evaluators may have varying opinions, and the evaluation criteria may lack consistency. Additionally, human evaluation can be time-consuming and expensive, especially for large-scale evaluations.

- *Limited reference data*: Some evaluation methods, such as BLEU or ROUGE, require reference data for comparison. However, obtaining high-quality reference data can be challenging, especially in scenarios where multiple acceptable responses exist or in open-ended tasks. Limited or biased reference data may not capture the full range of acceptable model outputs.

- *Lack of diversity metrics*: Existing evaluation methods often don’t capture the diversity and creativity of LLM outputs. That’s because metrics that only focus on accuracy and relevance overlook the importance of generating diverse and novel responses. Evaluating diversity in LLM outputs remains an ongoing research challenge.

- *Generalization to real-world scenarios*: Evaluation methods typically focus on specific benchmark datasets or tasks, which don’t fully reflect the challenges  of real-world applications. The evaluation on controlled datasets may not generalize well to diverse and dynamic contexts where LLMs are deployed.

- *Adversarial attacks*: LLMs can be susceptible to adversarial attacks such as manipulation of model predictions and data poisoning, where carefully crafted input can mislead or deceive the model. Existing evaluation methods often do not account for such attacks, and robustness evaluation remains an active area of research.

## **7. References**

[1] [Generative AI with Large Language Models Course of DeepLearning.AI on Coursera](https://www.coursera.org/learn/generative-ai-with-llms?utm_campaign=WebsiteCoursesGAIA&utm_medium=institutions&utm_source=deeplearning-ai)

[2] [Introduction to Large Language Models by Kumar Chandrakant on www.baeldung.com](https://www.baeldung.com/cs/large-language-models)

[3] [Large language model - Wikipedia](https://en.wikipedia.org/wiki/Large_language_model)
