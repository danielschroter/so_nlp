# Stack Overflow Tagging using NLP

This repository contains the code we developed for our project as part of the Applied Deep Learning for NLP seminar. Our goal was to use deep learning to assign tags to Stack Overflow questions (multiple tags per questions are possible and the norm).

We have developed several approaches to solving this problem and present them in notebooks within this project.

## The Models
We present the different model architectures in this section. Each of the three model architectures has an associated notebook that leads through the data processing and model training process.

### 1. Bag of Words Model
For the simplest of our models we simply encode word occurrences as a single one-hot vector for each question which represents occurrences of the top n (usually 1000) words in the entire corpus. All other tokens will not be used.

These embeddings are then fed into a simple sequential model with 2 fully connected layers, using relu as the activation function of the first and sigmoid for the output layer.

Data processing, model construction and training can be explored in [Bag_of_words.ipynb](Bag_of_words.ipynb).

### 2. Linear LSTM model for Title OR Body

This model is a constructed using the Keras Sequential API and can be considered linear in the sense that it maps a single input tensor to a single output tensor. For this model, we use [Facebook's fastText](https://fasttext.cc/) to train embeddings on the entire training corpus and use sequences of embedded question tokens as input for the LSTM layer. The last output of the LSTM layer is then passed to one or two dense layers to compute the final outputs (the last activation function is a sigmoid, the model therefore returns independent probabilities for each output label)

During training, model inputs are padded to the length of the question with the maximum number of tokens in the training dataset (usually 100 tokens, since we throw out bigger questions due to memory constraints during training). The padded model inputs are then masked by the [Masking layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking) of the model, so that padding elements are skipped during training (therefore not causing additional computation).

Data processing, model construction and training can be explored in [pipeline.ipynb](pipeline.ipynb). The pipeline can be configured to train on either question titles or bodies by setting the `use_titles` field in the first configuration cell accordingly.
### 3. Multi-Input LSTM for Title and Body

This model is a variation of the [Linear LSTM model for Title OR Body](#2.-linear-lstm-model-for-title-or-body) which takes in both question title and question body as inputs. These two inputs will be masked and passed into two separate LSTM layers (we use separate layers to allow them to individually discern between stylistic / syntactical differences in title and body). Their outputs are concatenated and processed as in the linear LSTM.

We provide a visualization of the model architecture below. For this visual example, we go with the following properties:
* batch size: 32
* sequence length: 50
* embedding size: 100
* lstm size (each): 64

![model architecture](graphics/title_body_model.svg)

## Comparison of Results


