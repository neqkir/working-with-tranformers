# working-with-tranformers

We use Google Colab to experiment with several models based on transformers (BERT, GPT-X, BART etc.) on various natural language processing tasks (text summarization, text generation, machine translation etc.). 

We first present below a summary of transformers-based models, from huggingface transformers library. 

We then present the details of each transformers exercise, model, task, data, reference paper, results.  

## summary of the models

https://huggingface.co/docs/transformers/model_summary 

Each transformer model (in huggingface) falls into one of the following categories 

* autoregressive-models

Autoregressive models are pretrained on the classic language modeling task: guess the next token having read all the previous ones. They correspond to the decoder of the original transformer model, and a mask is used on top of the full sentence so that the attention heads can only see what was before in the text, and not what’s after. Although those models can be fine-tuned and achieve great results on many tasks, the most natural application is text generation. A typical example of such models is GPT.

GPT, GPT-2, CTRL, transformer-XL, reformer, XLNet

* autoencoding-models

Autoencoding models are pretrained by corrupting the input tokens in some way and trying to reconstruct the original sentence. They correspond to the encoder of the original transformer model in the sense that they get access to the full inputs without any mask. Those models usually build a bidirectional representation of the whole sentence. They can be fine-tuned and achieve great results on many tasks such as text generation, but their most natural application is sentence classification or token classification. A typical example of such models is BERT.

Note that the only difference between autoregressive models and autoencoding models is in the way the model is pretrained. Therefore, the same architecture can be used for both autoregressive and autoencoding models. 

BERT, Albert, Roberta, DistilBert, ConvBert, XLM, XLM-Roberta, Flaubert, Electra, funnel transformer, longformer

* seq-to-seq-models

Sequence-to-sequence models use both the encoder and the decoder of the original transformer, either for translation tasks or by transforming other tasks to sequence-to-sequence problems. They can be fine-tuned to many tasks but their most natural applications are translation, summarization and question answering. The original transformer model is an example of such a model (only for translation), T5 is an example that can be fine-tuned on other tasks.

BART, Pegasus, T5, MarianMT, ProphetNet, MBart

* multimodal-models

Multimodal models mix text inputs with other kinds (e.g. images) and are more specific to a given task.

MMBT

* retrieval-based-models

Retrieval-based models use documents retrieval during (pre)training and inference for open-domain question answering, for example.

DPR, RAG

## codes

1. `training-bert-huggingface-transformers.ipynb`

Training huggingface BERT on the cc-news dataset which contains news articles from news sites all over the world. 

First, we train the WordPiece tokenizer. Next, we train BERT from scratch on the Masked Language Modeling (MLM) task, masking a certain percentage of the tokens in the sentence, and the model is trained to predict those masked words. Finally, model is applied to a few examples in text generation.

2. `gpt2-text-generators.ipynb`

Generating text with gpt-2 with several approaches for a generator : greedy search, beam search, sampling, top-K sampling, top-n-nucleus sampling.

Seeding with verses from the Proverbs, top-n nucleus sampling achieves very nice inventions

```
3: A satisfied soul loathes the honeycomb, But to a hungry soul every bitter thing is sweet.
That is not to say that I was not satisfied with my life. What was there to be pleased with? 
What kind of joy was there which the old man would have not taken away, In which an old soul would not have died, 
Or at least he would have died with a more perfect soul. And he would not have lived without an old soul, 
without those things which we
----------------------------------------------------------------------------------------------------
4: A satisfied soul loathes the honeycomb, But to a hungry soul every bitter thing is sweet. 
And when they say to the soul, "Do not eat this honey; I have eaten sweetness on a good feast for the last time";
the soul will answer, "Honey, it is bad on a good feast. I am now a bitter soul!"
This life consists of the following stages:
A full, clean heart of heart,
```

3. `bertsumabs-text-summarization.ipynb`

https://github.com/microsoft/nlp-recipes/blob/master/examples/text_summarization/abstractive_summarization_bertsumabs_cnndm.ipynb

Original paper (2019) : Text summarization with pretrained encoders https://arxiv.org/pdf/1908.08345.pdf

Pretrained models extend the idea of word embeddings learning contextual representation from large-scale corpora using a language modeling objective. With BERT each token from the input text is assigned three kinds of embeddings : 

> token embeddings indicate the meaning of each token, segmentation embeddings are used to discriminate between two sentences (e.g., during a sentence-pair classification > task) and position embeddingsindicate the position of each token within the text sequence. These three embeddings are summed to a single input vector xi and fed to a > bidirectional Transformer with multiple layers:

<img src="https://user-images.githubusercontent.com/89974426/148738594-6a2eb711-260d-4f37-b8a2-d94f042d8140.PNG" width=35% height=35%>    

The framework explores both extractive (summarize by extracting 'representative' important sentences from the original text) and abstractive (summarize by condensing the original text into a generated 'representative' text which abstracts or preserve important information) modeling paradigms. 

> We introduce a novel document-level encoder based on BERT which is able to express the semantics of a document and obtain representations for its sentences. Our extractive > model is built on top of this encoder by stacking several intersentence Transformer layers.

BERT is token-based while summarization needs manipulating sentence-level inputs. To this end BERT's architecture is modified as follows

<img src="https://user-images.githubusercontent.com/89974426/148779828-5b973ed6-4414-4fa9-ace7-2bab5a9ece4d.PNG" width=85% height=85%>    

Model evaluation : 

ROUGE-1 and ROUGE-2 evaluate informativeness, ROUGE-L fluency. Next the author evaluate the best learning rates for the optimizers (encoder and decoder have different optimizers). They assess the repartition across the whole text of the selected sentences and the proportion of novel n-grams, appearing in the summary but not the input text. 


4. `bart-text-summarization.ipynb`

https://github.com/sshleifer/blog_v2/blob/master/_notebooks/2020-03-12-bart.ipynb

We train and use BART for text summarization.

BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension https://arxiv.org/pdf/1910.13461.pdf 

BART starts from BERT,

* adding a causal decoder to BERT's bidirectional encoder architecture

* replace BERT's fill-in-the blank cloze task with a more complicated mix of pretraining tasks.

> (BART) can be seen as generalizing Bert (due to the bidirectional encoder) and GPT2 (with the left to right decoder).

"What information can be used use when predicting the token at position i" is controlled by an `attention_mask`. BERT has a fully visible mask, GPT a causal mask. BART (Seq2Seq) has a fully visible MASK for its encoder and a causal mask for its decoder. 

Pre-training masks spans of text, an example from the original paper.

<img src="https://user-images.githubusercontent.com/89974426/148036575-2e122b52-931a-45f8-b0c1-bf0c7aa69d2c.PNG" width=50% height=50%>    

Original document is A B C D E. the span `[C, D]` is masked before encoding, leaving the corrupted document `A _ B _ E` as input to the encoder.

The decoder (autogressive means "uses a causal mask") must reconstruct the original document, using the encoder's output and previous uncorrupted tokens.

Some results, a summary from BART

```
SomeSome of the possible possible things are different things that are different ones that are not the right 
ones that is not the one that is the right one that are the one of the one is the one for the one. 
Today the one in the world of the whole of the world.On this small a level, also known as the nanoscopic scale 
or nanoscale, what can be used to manipulate matter as small as atoms, small molecules, proteins, antibodies, 
and DNA bases that are less than 100 nanometers wide.For a few decades, scientists have been learning how to design 
molecules that can operate in the human body. The field holds a lot of potential when itcomes to handling questions
```

Admittedly not perfect, but a starting point for tuning around.

Some comparative performance results for text summarization

<img src="https://user-images.githubusercontent.com/89974426/148776092-391a951e-2ad4-43a9-ac79-7c9ecc94cd87.PNG" width=60% height=60%>    


5. `mmbt.ipynb`

"Supervised Multimodal Bitransformers for Classifying Images and Text" by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Ethan Perez and Davide Testuggine https://arxiv.org/abs/1909.02950




