# working-with-tranformers

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

3. `bart-text-summarization.ipynb`

> (BART) can be seen as generalizing Bert (due to the bidirectional encoder) and GPT2 (with the left to right decoder).

BART starts from BERT,

* adding a causal decoder to BERT's bidirectional encoder architecture

* replace BERT's fill-in-the blank cloze task with a more complicated mix of pretraining tasks.




