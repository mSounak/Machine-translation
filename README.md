# Machine-translation

## Introduction

This project is focused on self learning purpose. In this project, I have created a Sequence-to-Sequence translation model, that helps to translate english sentences into german. This is a first iteration and can be improved. 

## Dataset and Preprocessing

I have used the dataset from [Anki](https://www.manythings.org/anki/deu-eng.zip). The dataset contains a text file with english sentences and german sentences.
Data looks like this
`English + TAB + German + TAB + Attribution`

### Preprocessing

For preprocessing the text, I have added `[start]` and `[end]` token at the beginning and the end of the german sentences, respectively. And removed the punctuations from the english and the german sentences. Then used TextVectorization layer from tensorflow to convert the text into a sequence of integers.

## Model

Created a sequence-to-sequence Transformer model that consist of a `TransformerEncoder`, `PostionalEmbedding` and `TransformerDecoder`.
Let's see how each layer works in our model.
* `TransformerEncoder` : Transformer Encoder takes the embedding of the english sentences then pass it to the Multihead Attention layer. The output of the Multihead Attention layer is then added with the embedded sentence and we do a layer normalization on them. We then pass to the Feed Forward layer. The output of the Feed Forward layer is then passed to the `TransformerDecoder`.

* `TransformerDecoder` : Transformer Decoder takes the embedding of the german sentences then pass it to the Multihead Attention layer. The output of the Multihead Attention layer is then added with the embedded german sentences and we do a layer normalization on them. We then pass the layerNorm output as query to the Multihead Attention and the encoder output as key and value. We then add the output of the Multihead Attention layer to the output of the 1st Multihead Attention layer. We then pass to the Feed Forward layer. The output of the Feed Forward layer is then passed through a dense layer and then we perform softmax on the output.

* `PostionalEmbedding` : This layer is used to create positional embedding. Attention layers see their input as a set of vectors, with no sequential order. Since, the model doesn't have any reccurent or convolution layers. Because of this positional encoding is added to give the model some information about the relative position of the tokens in the sentence.

![transformer_model](https://miro.medium.com/max/860/1*InsTuWpZTYm0kwi8ovIMAQ.png)

## Inference

We simply feed the into the model the vectorized english sentence as well as the target token "[start]", then we repeatedly generate the next token, until we hit the token "[end]".


## Future Plans
- Add support for Indic languages
- Create a better preprocessing pipeline

## Deployment
I have deployed this project in AWS but due to lack of free credits shifted it to Alibaba cloud, [click here to give it a try](http://147.139.35.89:8501/)

![nmt](https://i.imgur.com/7bHalBG.png)



<hr>

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/>

<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" />
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

