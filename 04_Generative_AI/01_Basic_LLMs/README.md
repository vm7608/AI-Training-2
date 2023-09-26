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

The v`anishing gradient problem is encountered in ANN` using gradient-based learning methods with backpropagation. In such methods, during each iteration of training, the weights receive an update proportional to the partial derivative of the error function concerning the current weight.

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

Maybe sometime model performs well at the first try, you'll frequently encounter situations where the model doesn't produce the outcome that you want on the first try. You may have to revise the language in your prompt or the way that it's written several times to get the model to behave in the way that you want. This work to develop and improve the prompt is known as `prompt engineering`.

`Prompt engineering` refers to the process of carefully crafting textual prompts to elicit targeted behaviors from large language models (LLMs). The goal of prompt engineering is developing textual contexts that maximize the benefits of language models while mitigating risks like toxic, factually incorrect, or unaligned responses. It's an important technique for applying LLMs safely and ensuring user experiences meet expectations. Proper prompt design upfront helps avoid potential downstream issues.

But one powerful strategy to get the model to produce better outcomes is to include examples of the task that you want the model to carry out inside the prompt. Providing examples inside the context window is called `in-context learning`. There are the following types of in-context learning:

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

A drawback of in-context learning is that out context window have a limit amount of token so that we can't include too many examples. Generally, if you find that your model isn't performing well when including five or six examples, you should try fine-tuning your model instead. Fine-tuning performs additional training on the model using new data to make it more capable of the task you want it to perform.

### **3.2. Generative configuration parameters for inference**

In LLMs model, we have some parameters that we can use to control the model's behavior. These parameters are called `generative configuration parameters`. Each model exposes a set of configuration parameters that can influence the model's output during inference. Note that these are different than the training parameters which are learned during training time. Instead, these configuration parameters are invoked at inference time and give you control over things like the maximum number of tokens in the completion, and how creative the output is.

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

If you want to generate text that's more natural, more creative and avoids repeating words, you need to use some other controls. `Random sampling` is the easiest way to introduce some variability. Instead of selecting the most probable word every time with random sampling, the model chooses an output word at random using the probability distribution to weight the selection. For example, in the illustration below, the word banana has a probability score of 0.02. With random sampling, this equates to a 2% chance that this word will be selected. By using this sampling technique, we reduce the likelihood that words will be repeated. However, depending on the setting, there is a possibility that the output may be too creative, producing words that cause the generation to wander off into topics or words that just don't make sense. Note that in some implementations, you may need to disable greedy and enable random sampling explicitly.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53213684897_2ce6f82b9a_o.png" >
  <br>
  <i>Greedy vs Random sampling</i>
</p>

Here we have `Top k`, `Top p` and `Temperature`, these are three parameters that can be used to control the model's creativity.

`Top k` focuses generation on a small subset of most likely tokens. At each step, the model considers only the k most likely next tokens according to the predicted probabilities (the top k highest probability tokens). The model then selects from these k options using the probability weighting. This method can help the model have some randomness while preventing the selection of highly improbable completion words. This in turn makes your text generation more likely to sound reasonable and to make sense.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53213684887_f04280c5b0_o.png" >
  <br>
  <i>Top k</i>
</p>

Alternatively, you can use the `Top p` setting to limit the random sampling to the predictions whose combined probabilities do not exceed p. For example, if you set p to equal 0.3, We add up probabilities from most to least likely until we cross the 0.3 threshold. The options are cake and donut since their probabilities of 0.2 and 0.1 add up to 0.3. The model then uses the random probability weighting method to choose from these tokens.

<p align="center">
  <img src="https://live.staticflickr.com/65535/53215079035_2a65f1ebd9_o.png" >
  <br>
  <i>Top p</i>
</p>

One more parameter that you can use to control the randomness of the model output is known as `temperature`. This parameter influences the shape of the probability distribution that the model calculates for the next token. The higher the temperature, the higher the randomness, and the lower the temperature, the lower the randomness. The temperature value is a scaling factor that's applied within the final softmax layer of the model that impacts the shape of the probability distribution of the next token. In contrast to the top k and top p parameters, changing the temperature actually alters the predictions that the model will make.

- If you choose a low value of temperature (less than one), the resulting probability distribution from the softmax layer is more strongly peaked with the probability being concentrated in a smaller number of words. Most of the probability here is concentrated on the word cake. The model will select from this distribution using random sampling and the resulting text will be less random and will more closely follow the most likely word sequences that the model learned during training.
- If instead you set the temperature to a higher value (greater than one), then the model will calculate a broader flatter probability distribution for the next token. Notice that in contrast to the blue bars, the probability is more evenly spread across the tokens. This leads the model to generate text with a higher degree of randomness and more variability in the output compared to a cool temperature setting. This can help you generate text that sounds more creative.
- If you leave the temperature value equal to one, this will leave the softmax function as default and the unaltered probability distribution will be used.

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
  - If the model does not perform as well as we need, even with one or a few short inference, and in that case, we can try fine-tuning your model.
  - As models become more capable, it's important to ensure that they behave well and in a way that is aligned with human preferences in deployment. Here, we will perform an additional fine-tuning technique called reinforcement learning with human feedback, which can help to make sure that model behaves well.
  - An important aspect of all of these techniques is evaluation. We should use some metrics and benchmarks to determine how well our model is performing or how well aligned it is to our preferences.

- `Application integration`: Finally, when we've got a model that is good perform and well aligned, we can deploy it into an infrastructure and integrate an application. At this stage, an important step is to optimize your model for deployment to ensure of efficient compute resources and good experience for the users.

### **3.4. Pre-training large language models**

#### **3.4.1. Pre-training LLMs at a high level**

The initial training process for LLMs is often referred to as pre-training. LLMs encode a deep statistical representation of language. This understanding is developed during the models pre-training phase when the model learns from vast amounts of unstructured textual data. This can be gigabytes, terabytes, and even petabytes of text. This data is pulled from many sources, including scrapes off the Internet and corpora of texts that have been assembled specifically for training language models.

In this self-supervised learning step, the model internalizes the patterns and structures present in the language. These patterns then enable the model to complete its training objective, which depends on the architecture of the model. During pre-training, the model weights get updated to minimize the loss of the training objective. The encoder generates an embedding or vector representation for each token. Pre-training also requires a large amount of compute and the use of GPUs. 

Note, when you scrape training data from public sites such as the Internet, you often need to process the data to increase quality, address bias, and remove other harmful content. As a result of this data quality curation, often only 1-3% of tokens are used for pre-training.

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

`Encoder-only models` are also known as `Autoencoding models`, and they are pre-trained using masked language modeling. Here, tokens in the input sequence or randomly mask, and the training objective is to predict the mask tokens in order to reconstruct the original sentence. This is also called a denoising objective. Autoencoding models spilled bi-directional representations of the input sequence, meaning that the model has an understanding of the full context of a token and not just of the words that come before. Encoder-only models are ideally suited to task that benefit from this bi-directional contexts. You can use them to carry out sentence classification tasks, for example, sentiment analysis or token-level tasks like named entity recognition or word classification. Some well-known examples of an autoencoder model are BERT and RoBERTa.

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

The final variation of the transformer model is the sequence-to-sequence model that uses both the encoder and decoder parts off the original transformer architecture. The exact details of the pre-training objective vary from model to model. A popular sequence-to-sequence model T5, pre-trains the encoder using span corruption, which masks random sequences of input tokens. Those mass sequences are then replaced with a unique `Sentinel token`, shown here as x. `Sentinel tokens` are special tokens added to the vocabulary, but do not correspond to any actual word from the input text. The decoder is then tasked with reconstructing the mask token sequences auto-regressively. The output is the Sentinel token followed by the predicted tokens. You can use sequence-to-sequence models for translation, summarization, and question-answering. They are generally useful in cases where you have a body of texts as both input and output.

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

### **4.1. Instruction Tuning**

#### **4.1.1. Instruction fine-tuning for a single task**

#### **4.1.2. Instruction fine-tuning for multiple tasks**

### **4.2. Parameter efficient fine-tuning (PEFT)**

#### **4.2.1. PEFT techniques 1: LORA**

#### **4.2.2. PEFT techniques 2: Soft Prompt Tuning**

## **5. Reinforcement Learning From Human Feedback (RLHF)**

## **6. Evaluating LLMs**

### **6.1. Intrinsic Methods**

### **6.2. Extrinsic Methods**

### **6.3. Benchmarking**

### **6.4. Limitations and potential biases**

## **7. References**

- To be defined
