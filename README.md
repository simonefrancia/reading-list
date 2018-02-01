# Reading list

This repository contains my reading list for deep learning research. Topics
are mainly: relation extraction, reading comprehension, neural network
architectures, classification, neural machine translation and many more.

# 2018

## Deep contextualized word representations

Authors: Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner,
         Christopher Clark, Kenton Lee, Luke Zettlemoyer

Abstract:

> We introduce a new type of deep contextualized word representation that models
> both (1) complex characteristics of word use (e.g., syntax and semantics), and
> (2) how these uses vary across linguistic contexts (i.e., to model polysemy).
> Our word vectors are learned functions of the internal states of a deep
> bidirectional language model (biLM), which is pretrained on a large text
> corpus. We show that these representations can be easily added to existing
> models and significantly improve the state of the art across six challenging
> NLP problems, including question answering, textual entailment and sentiment
> analysis.  We also present an analysis showing that exposing the deep internals
> of the pretrained network is crucial, allowing downstream models to mix
> different types of semi-supervision signals.

Material(s): [[ICLR 2018](https://openreview.net/forum?id=S1p31z-Ab)]

## Fine-tuned Language Models for Text Classification

Authors: Jeremy Howard, Sebastian Ruder

Abstract:

> Transfer learning has revolutionized computer vision, but existing
> approaches in NLP still require task-specific modifications and training from
> scratch. We propose Fine-tuned Language Models (FitLaM), an effective transfer
> learning method that can be applied to any task in NLP, and introduce
> techniques that are key for fine-tuning a state-of-the-art language model. Our
> method significantly outperforms the state-of-the-art on five text
> classification tasks, reducing the error by 18-24% on the majority of datasets.
> We open-source our pretrained models and code to enable adoption by the
> community.

Material(s): [[arxiv](https://arxiv.org/abs/1801.06146)]

## Towards Neural Phrase-based Machine Translation

Authors: Po-Sen Huang, Chong Wang, Sitao Huang, Dengyong Zhou, Li Deng

Abstract:

> In this paper, we present Neural Phrase-based Machine Translation (NPMT). Our
> method explicitly models the phrase structures in output sequences using
> Sleep-WAke Networks (SWAN), a recently proposed segmentation-based sequence
> modeling method. To mitigate the monotonic alignment requirement of SWAN, we
> introduce a new layer to perform (soft) local reordering of input sequences.
> Different from existing neural machine translation (NMT) approaches, NPMT does
> not use attention-based decoding mechanisms. Instead, it directly outputs
> phrases in a sequential order and can decode in linear time. Our experiments
> show that NPMT achieves superior performances on IWSLT 2014
> German-English/English-German and IWSLT 2015 English-Vietnamese machine
> translation tasks compared with strong NMT baselines. We also observe that our
> method produces meaningful phrases in output languages.

Material(s): [[arxiv](https://arxiv.org/abs/1706.05565)] - [[GitHub](https://github.com/posenhuang/NPMT)]
