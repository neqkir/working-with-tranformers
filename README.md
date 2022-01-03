# working-with-tranformers

1. `training-bert-huggingface-transformers.ipynb`

Training huggingface BERT on the cc-news dataset which contains news articles from news sites all over the world. 

First, we train the WordPiece tokenizer. Next, we train BERT from scratch on the Masked Language Modeling (MLM) task, masking a certain percentage of the tokens in the sentence, and the model is trained to predict those masked words.  

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
