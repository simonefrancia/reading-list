# Reading list

This repository contains my reading list for deep learning research. Topics
are mainly: relation extraction, reading comprehension, neural network
architectures, classification, neural machine translation and many more.

# 2019

## ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators

Authors: (currently) Anonymous

Abstract:

> While masked language modeling (MLM) pre-training methods such as BERT produce
> excellent results on downstream NLP tasks, they require large amounts of
> compute to be effective. These approaches corrupt the input by replacing some
> tokens with [MASK] and then train a model to reconstruct the original tokens.
> As an alternative, we propose a more sample-efficient pre-training task called
> replaced token detection. Instead of masking the input, our approach corrupts
> it by replacing some input tokens with plausible alternatives sampled from a
> small generator network. Then, instead of training a model that predicts the
> original identities of the corrupted tokens, we train a discriminative model
> that predicts whether each token in the corrupted input was replaced by a
> generator sample or not. Thorough experiments demonstrate this new pre-training
> task is more efficient than MLM because the model learns from all input tokens
> rather than just the small subset that was masked out. As a result, the
> contextual representations learned by our approach substantially outperform the
> ones learned by methods such as BERT and XLNet given the same model size, data,
> and compute. The gains are particularly strong for small models; for example,
> we train a model on one GPU for 4 days that outperforms GPT (trained using 30x
> more compute) on the GLUE natural language understanding benchmark.
> Our approach also works well at scale, where we match the performance of
> RoBERTa, the current state-of-the-art pre-trained transformer, while using
> less than 1/4 of the compute.

Materials(s): [[OpenReview Paper](https://openreview.net/forum?id=r1xMH1BtvB)]

## ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

Authors: (currently) Anonymous

Abstract:

> Increasing model size when pretraining natural language representations often
> results in improved performance on downstream tasks. However, at some point
> further model increases become harder due to GPU/TPU memory limitations,
> longer training times, and unexpected model degradation. To address these
> problems, we present two parameter-reduction techniques to lower memory
> consumption and increase the training speed of BERT. Comprehensive empirical
> evidence shows that our proposed methods lead to models that scale much better
> compared to the original BERT. We also use a self-supervised loss that focuses
> on modeling inter-sentence coherence, and show it consistently helps downstream
> tasks with multi-sentence inputs. As a result, our best model establishes new
> state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having
> fewer parameters compared to BERT-large. 

Materials(s): [[OpenReview Paper](https://openreview.net/forum?id=H1eA7AEtvS)]

## Small and Practical BERT Models for Sequence Labeling

Authors: Henry Tsai, Jason Riesa, Melvin Johnson, Naveen Arivazhagan, Xin Li, Amelia Archer

Abstract:

> We propose a practical scheme to train a single multilingual sequence labeling
> model that yields state of the art results and is small and fast enough to run
> on a single CPU. Starting from a public multilingual BERT checkpoint, our final
> model is 6x smaller and 27x faster, and has higher accuracy than a
> state-of-the-art multilingual baseline. We show that our model especially
> outperforms on low-resource languages, and works on codemixed input text
> without being explicitly trained on codemixed examples. We showcase the
> effectiveness of our method by reporting on part-of-speech tagging and
> morphological prediction on 70 treebanks and 48 languages.

Materials(s): [[Paper](https://arxiv.org/abs/1909.00100)]

## Distilling Task-Specific Knowledge from BERT into Simple Neural Networks

Authors: Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, Jimmy Lin

Abstract:

> In the natural language processing literature, neural networks are becoming
> increasingly deeper and complex. The recent poster child of this trend is the
> deep language representation model, which includes BERT, ELMo, and GPT. These
> developments have led to the conviction that previous-generation, shallower
> neural networks for language understanding are obsolete. In this paper,
> however, we demonstrate that rudimentary, lightweight neural networks can
> still be made competitive without architecture changes, external training data,
> or additional input features. We propose to distill knowledge from BERT,
> a state-of-the-art language representation model, into a single-layer BiLSTM,
> as well as its siamese counterpart for sentence-pair tasks. Across multiple
> datasets in paraphrasing, natural language inference, and sentiment
> classification, we achieve comparable results with ELMo, while using roughly
> 100 times fewer parameters and 15 times less inference time.

Materials(s): [[Paper](https://arxiv.org/abs/1903.12136)]

## Hierarchically-Refined Label Attention Network for Sequence Labeling

Authors: Leyang Cui, Yue Zhang

Abstract:

> CRF has been used as a powerful model for statistical sequence labeling.
> For neural sequence labeling, however, BiLSTM-CRF does not always lead to
> better results compared with BiLSTM-softmax local classification. This can be
> because the simple Markov label transition model of CRF does not give much
> information gain over strong neural encoding. For better representing label
> sequences, we investigate a hierarchically-refined label attention network,
> which explicitly leverages label embeddings and captures potential long-term
> label dependency by giving each word incrementally refined label distributions
> with hierarchical attention. Results on POS tagging, NER and CCG supertagging
> show that the proposed model not only improves the overall tagging accuracy
> with similar number of parameters, but also significantly speeds up the
> training and testing compared to BiLSTM-CRF.

Materials(s): [[Paper](https://arxiv.org/abs/1908.08676)] - [[GitHub repo](https://github.com/Nealcly/LAN)]

## RoBERTa: A Robustly Optimized BERT Pretraining Approach

Authors: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov

Abstract:

> Language model pretraining has led to significant performance gains but careful
> comparison between different approaches is challenging. Training is
> computationally expensive, often done on private datasets of different sizes,
> and, as we will show, hyperparameter choices have significant impact on the
> final results. We present a replication study of BERT pretraining
> (Devlin et al., 2019) that carefully measures the impact of many key
> hyperparameters and training data size. We find that BERT was significantly
> undertrained, and can match or exceed the performance of every model published
> after it. Our best model achieves state-of-the-art results on GLUE, RACE and
> SQuAD. These results highlight the importance of previously overlooked design
> choices, and raise questions about the source of recently reported improvements.
> We release our models and code.

Materials(s): [[Paper](https://arxiv.org/abs/1907.11692)] - [[`fairseq` GitHub repo](https://github.com/pytorch/fairseq)]

## Sequence Tagging with Contextual and Non-Contextual Subword Representations: A Multilingual Evaluation

Authors: Benjamin Heinzerling, Michael Strube

Abstract:

> Pretrained contextual and non-contextual subword embeddings have become
> available in over 250 languages, allowing massively multilingual NLP.
> However, while there is no dearth of pretrained embeddings, the distinct lack
> of systematic evaluations makes it difficult for practitioners to choose
> between them. In this work, we conduct an extensive evaluation comparing
> non-contextual subword embeddings, namely FastText and BPEmb, and a contextual
> representation method, namely BERT, on multilingual named entity recognition
> and part-of-speech tagging. We find that overall, a combination of BERT,
> BPEmb, and character representations works best across languages and tasks.
> A more detailed analysis reveals different strengths and weaknesses:
> Multilingual BERT performs well in medium- to high-resource languages, but is
> outperformed by non-contextual subword embeddings in a low-resource setting.

Materials(s): [[Paper](https://arxiv.org/abs/1906.01569)]

Comments: What a great multi-lingual evaluation on over 250 languages! New SOTA
for PoS tagging on Universal Dependencies. The *MultiBPEmb* model will be
included into [bpemb](https://github.com/bheinzerling/bpemb) soon!

## Adaptive Attention Span in Transformers

Authors: Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, Armand Joulin

Abstract:

> We propose a novel self-attention mechanism that can learn its optimal attention
> span. This allows us to extend significantly the maximum context size used in
> Transformer, while maintaining control over their memory footprint and
> computational time. We show the effectiveness of our approach on the task of
> character level language modeling, where we achieve state-of-the-art
> performances on text8 and enwiki8 by using a maximum context of 8k characters.

Materials(s): [[Paper](https://arxiv.org/abs/1905.07799)]

## Unified Language Model Pre-training for Natural Language Understanding and Generation

Authors: Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon

Abstract:

> This paper presents a new Unified pre-trained Language Model (UniLM) that can
> be fine-tuned for both natural language understanding and generation tasks.
> The model is pre-trained using three types of language modeling objectives:
> unidirectional (both left-to-right and right-to-left), bidirectional, and
> sequence-to-sequence prediction. The unified modeling is achieved by employing
> a shared Transformer network and utilizing specific self-attention masks to
> control what context the prediction conditions on. We can fine-tune UniLM as a
> unidirectional decoder, a bidirectional encoder, or a sequence-to-sequence
> model to support various downstream natural language understanding and
> generation tasks. UniLM compares favorably with BERT on the GLUE benchmark,
> and the SQuAD 2.0 and CoQA question answering tasks. Moreover, our model
> achieves new state-of-the-art results on three natural language generation
> tasks, including improving the CNN/DailyMail abstractive summarization ROUGE-L
> to 40.63 (2.16 absolute improvement), pushing the CoQA generative question
> answering F1 score to 82.5 (37.1 absolute improvement), and the SQuAD question
> generation BLEU-4 to 22.88 (6.50 absolute improvement). 

Materials(s): [[Paper](https://arxiv.org/abs/1905.03197)]

## Rare Words: A Major Problem for Contextualized Embeddings And How to Fix it by Attentive Mimicking

Authors: Timo Schick, Hinrich Schütze

Abstract:

> Pretraining deep neural network architectures with a language modeling objective
> has brought large improvements for many natural language processing tasks.
> Exemplified by BERT, a recently proposed such architecture, we demonstrate that
> despite being trained on huge amounts of data, deep language models still
> struggle to understand rare words. To fix this problem, we adapt
> Attentive Mimicking, a method that was designed to explicitly learn embeddings
> for rare words, to deep language models. In order to make this possible, we
> introduce one-token approximation, a procedure that enables us to use Attentive
> Mimicking even when the underlying language model uses subword-based
> tokenization, i.e., it does not assign embeddings to all words. To evaluate our
> method, we create a novel dataset that tests the ability of language models to
> capture semantic properties of words without any task-specific fine-tuning.
> Using this dataset, we show that adding our adapted version of Attentive
> Mimicking to BERT does indeed substantially improve its understanding of rare
> words.

Materials(s): [[Paper](https://arxiv.org/abs/1904.06707)] - [[GitHub implementation of the form-context model](https://github.com/timoschick/form-context-model)]

## Pooled Contextualized Embeddings for Named Entity Recognition

Authors: Alan Akbik, Tanja Bergmann, Roland Vollgraf

Abstract:

> Contextual string embeddings are a recent typeof contextualized word embedding
> that were shown to yield state-of-the-art results when utilized in a range of
> sequence labeling tasks. They are based on character-level language models which
> treat text as distributions over characters and are capable of generating
> embeddings for any string of characters within any textual context. However,
> such purely character-based approaches struggle to produce meaningful embeddings
> if a rare string is used in a underspecified context. To address this drawback,
> we propose a method in which we dynamically aggregate contextualized embeddings
> of each unique string that we encounter. We then use a pooling operation to
> distill a global word representation from all contextualized instances. We
> evaluate these pooled contextualized embeddings on common named entity
> recognition (NER) tasks such as CoNLL-03 and WNUT and show that our approach
> significantly improves the state-of-the-art for NER. We make all code and
> pre-trained models available to the research community for use and reproduction.

Materials(s): [[Paper](http://alanakbik.github.io/papers/naacl2019_embeddings.pdf)] - [[GitHub](https://github.com/zalandoresearch/flair)]

## SciBERT: Pretrained Contextualized Embeddings for Scientific Text

Authors: Iz Beltagy, Arman Cohan, Kyle Lo

Abstract:

> Obtaining large-scale annotated data for NLP tasks in the scientific domain
> is challenging and expensive. We release SciBERT, a pretrained contextualized
> embedding model based on BERT (Devlin et al., 2018) to address the lack of
> high-quality, large-scale labeled scientific data. SciBERT leverages
> unsupervised pretraining on a large multi-domain corpus of scientific
> publications to improve performance on downstream scientific NLP tasks.
> We evaluate on a suite of tasks including sequence tagging, sentence
> classification and dependency parsing, with datasets from a variety of
> scientific domains. We demonstrate statistically significant improvements
> over BERT and achieve new state-of-the-art results on several of these tasks.

Materials(s): [[Paper](https://arxiv.org/abs/1903.10676)] - [[GitHub](https://github.com/allenai/scibert)]

## Linguistic Knowledge and Transferability of Contextual Representations

Authors: Nelson F. Liu, Matt Gardner, Yonatan Belinkov, Matthew Peters, Noah A. Smith

Abstract:

> Contextual word representations derived from large-scale neural language models
> are successful across a diverse set of NLP tasks, suggesting that they encode
> useful and transferable features of language. To shed light on the linguistic
> knowledge they capture, we study the representations produced by several recent
> pretrained contextualizers (variants of ELMo, the OpenAI transformer LM, and BERT)
> with a suite of sixteen diverse probing tasks. We find that linear models trained
> on top of frozen contextual representations are competitive with state-of-the-art
> task-specific models in many cases, but fail on tasks requiring fine-grained
> linguistic knowledge (e.g., conjunct identification). To investigate the
> transferability of contextual word representations, we quantify differences in the
> transferability of individual layers within contextualizers, especially between
> RNNs and transformers. For instance, higher layers of RNNs are more task-specific,
> while transformer layers do not exhibit the same monotonic trend. In addition, to
> better understand what makes contextual word representations transferable, we
> compare language model pretraining with eleven supervised pretraining tasks.
> For any given task, pretraining on a closely related task yields better
> performance than language model pretraining (which is better on average) when the
> pretraining dataset is fixed. However, language model pretraining on more data
> gives the best results. 

Materials(s): [[Paper](https://arxiv.org/abs/1903.08855)]

## Cloze-driven Pretraining of Self-attention Networks

Authors: Alexei Baevski, Sergey Edunov, Yinhan Liu, Luke Zettlemoyer, Michael Auli

Abstract:

> We present a new approach for pretraining a bi-directional transformer model
> that provides significant performance gains across a variety of language
> understanding problems. Our model solves a cloze-style word reconstruction
> task, where each word is ablated and must be predicted given the rest of the
> text. Experiments demonstrate large performance gains on GLUE and new state of
> the art results on NER as well as constituency parsing benchmarks, consistent
> with the concurrently introduced BERT model. We also present a detailed analysis
> of a number of factors that contribute to effective pretraining, including data
> domain and size, model capacity, and variations on the cloze objective. 

Materials(s): [[Paper](https://arxiv.org/abs/1903.07785)]

Comments: New SOTA e.g. for NER. With the large CNN model (+ fine-tuning) a
F1-Score of 93.5% can be achieved.

## To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks

Authors: Matthew Peters, Sebastian Ruder, Noah A. Smith

Abstract:

> While most previous work has focused on different pretraining objectives and
> architectures for transfer learning, we ask how to best adapt the pretrained
> model to a given target task. We focus on the two most common forms of
> adaptation, feature extraction (where the pretrained weights are frozen), and
> directly fine-tuning the pretrained model. Our empirical results across diverse
> NLP tasks with two state-of-the-art models show that the relative performance
> of fine-tuning vs. feature extraction depends on the similarity of the
> pretraining and target tasks. We explore possible explanations for this
> finding and provide a set of adaptation guidelines for the NLP practitioner.

Materials(s): [[Paper](https://arxiv.org/abs/1903.05987)]

## Parameter-Efficient Transfer Learning for NLP

Authors: Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly

Abstract:

> Fine-tuning large pre-trained models is an effective transfer mechanism in NLP.
> However, in the presence of many downstream tasks, fine-tuning is parameter
> inefficient: an entire new model is required for every task. As an alternative,
> we propose transfer with adapter modules. Adapter modules yield a compact and
> extensible model; they add only a few trainable parameters per task, and new
> tasks can be added without revisiting previous ones. The parameters of the
> original network remain fixed, yielding a high degree of parameter sharing.
> To demonstrate adapter's effectiveness, we transfer the recently proposed BERT
> Transformer model to 26 diverse text classification tasks, including the GLUE
> benchmark. Adapters attain near state-of-the-art performance, whilst adding
> only a few parameters per task. On GLUE, we attain within 0.4% of the
> performance of full fine-tuning, adding only 3.6% parameters per task.
> By contrast, fine-tuning trains 100% of the parameters per task. 

Materials(s): [[Paper](https://arxiv.org/abs/1902.00751)]

## Pay Less Attention with Lightweight and Dynamic Convolutions

Authors: Felix Wu, Angela Fan, Alexei Baevski, Yann N. Dauphin, Michael Auli

Abstract:

> Self-attention is a useful mechanism to build generative models for
> language and images. It determines the importance of context elements
> by comparing each element to the current time step. In this paper, we
> show that a very lightweight convolution can perform competitively to
> the best reported self-attention results. Next, we introduce dynamic
> convolutions which are simpler and more efficient than self-attention.
> We predict separate convolution kernels based solely on the current
> time-step in order to determine the importance of context elements. The
> number of operations required by this approach scales linearly in the
> input length, whereas self-attention is quadratic. Experiments on
> large-scale machine translation, language modeling and abstractive
> summarization show that dynamic convolutions improve over strong
> self-attention models. On the WMT'14 English-German test set dynamic
> convolutions achieve a new state of the art of 29.7 BLEU.

Materials(s): [[Paper](https://arxiv.org/abs/1901.10430)] - [[Fairseq Implementation](https://github.com/pytorch/fairseq/pull/473)]

Comments: Well written paper, strong baselines. I'll definitely try it
out and report results in my
[English-Macedonian](https://github.com/stefan-it/nmt-en-mk)
and [English-Vietnamese](https://github.com/stefan-it/nmt-en-vi) repos.
Result for language modeling on Billion Word test set is lower than
the reported one in the Transformer-XL paper.

## Cross-lingual Language Model Pretraining

Authors: Guillaume Lample, Alexis Conneau

Abstract:

> Recent studies have demonstrated the efficiency of generative
> pretraining for English natural language understanding. In this work,
> we extend this approach to multiple languages and show the
> effectiveness of cross-lingual pretraining. We propose two methods to
> learn cross-lingual language models (XLMs): one unsupervised that only
> relies on monolingual data, and one supervised that leverages parallel
> data with a new cross-lingual language model objective. We obtain
> state-of-the-art results on cross-lingual classification, unsupervised
> and supervised machine translation. On XNLI, our approach pushes the
> state of the art by an absolute gain of 4.9% accuracy. On unsupervised
> machine translation, we obtain 34.3 BLEU on WMT'16 German-English,
> improving the previous state of the art by more than 9 BLEU. On
> supervised machine translation, we obtain a new state of the art of
> 38.5 BLEU on WMT'16 Romanian-English, outperforming the previous best
> approach by more than 4 BLEU. Our code and pretrained models will be
> made publicly available.

Materials(s): [[Paper](https://arxiv.org/abs/1901.07291)]

## Transformer-XL: Language Modeling with Longer-Term Dependency

Authors: Zihang Dai, Zhilin Yang, Yiming Yang, William W. Cohen, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov

> We propose a novel neural architecture, Transformer-XL, for modeling
> longer-term dependency. To address the limitation of fixed-length
> contexts, we introduce a notion of recurrence by reusing the
> representations from the history. Empirically, we show state-of-the-art
> (SoTA) results on both word-level and character-level language modeling
> datasets, including WikiText-103, One Billion Word, Penn Treebank, and
> enwiki8. Notably, we improve the SoTA results from 1.06 to 0.99 in bpc
> on enwiki8, from 33.0 to 18.9 in perplexity on WikiText-103, and from
> 28.0 to 23.5 in perplexity on One Billion Word. Performance improves
> when the attention length increases during evaluation, and our best
> model attends to up to 1,600 words and 3,800 characters. To quantify
> the effective length of dependency, we devise a new metric and show
> that on WikiText-103 Transformer-XL manages to model dependency that is
> about 80% longer than recurrent networks and 450% longer than
> Transformer. Moreover, Transformer-XL is up to 1,800+ times faster than
> vanilla Transformer during evaluation.

Materials(s): [[Paper](https://openreview.net/forum?id=HJePno0cYm)] - [[GitHub](https://github.com/kimiyoung/transformer-xl)]

# 2018

## DTMT: A Novel Deep Transition Architecture for Neural Machine Translation

Abstract:

> Past years have witnessed rapid developments in Neural Machine
> Translation (NMT). Most recently, with advanced modeling and training
> techniques, the RNN-based NMT (RNMT) has shown its potential strength,
> even compared with the well-known Transformer (self-attentional) model.
> Although the RNMT model can possess very deep architectures through
> stacking layers, the transition depth between consecutive hidden states
> along the sequential axis is still shallow. In this paper, we further
> enhance the RNN-based NMT through increasing the transition depth
> between consecutive hidden states and build a novel Deep Transition
> RNN-based Architecture for Neural Machine Translation, named DTMT. This
> model enhances the hidden-to-hidden transition with multiple non-linear
> transformations, as well as maintains a linear transformation path
> throughout this deep transition by the well-designed linear
> transformation mechanism to alleviate the gradient vanishing problem.
> Experiments show that with the specially designed deep transition
> modules, our DTMT can achieve remarkable improvements on translation
> quality. Experimental results on Chinese->English translation task show
> that DTMT can outperform the Transformer model by +2.09 BLEU points and
> achieve the best results ever reported in the same dataset. On WMT14
> English->German and English->French translation tasks, DTMT shows
> superior quality to the state-of-the-art NMT systems, including the
> Transformer and the RNMT+.

Materials(s): [[Paper](https://arxiv.org/abs/1812.07807)]

## Semi-Supervised Sequence Modeling with Cross-View Training

Abstract:

> Unsupervised representation learning algorithms such as word2vec and
> ELMo improve the accuracy of many supervised NLP models, mainly because
> they can take advantage of large amounts of unlabeled text. However,
> the supervised models only learn from task-specific labeled data during
> the main training phase. We therefore propose Cross-View Training
> (CVT), a semi-supervised learning algorithm that improves the
> representations of a Bi-LSTM sentence encoder using a mix of labeled
> and unlabeled data. On labeled examples, standard supervised learning
> is used. On unlabeled examples, CVT teaches auxiliary prediction
> modules that see restricted views of the input (e.g., only part of a
> sentence) to match the predictions of the full model seeing the whole
> input. Since the auxiliary modules and the full model share
> intermediate representations, this in turn improves the full model.
> Moreover, we show that CVT is particularly effective when combined with
> multi-task learning. We evaluate CVT on five sequence tagging tasks,
> machine translation, and dependency parsing, achieving state-of-the-art
> results.

Material(s): [[Paper](https://arxiv.org/abs/1809.08370)]

## A Named Entity Recognition Shootout for German

Authors: Martin Riedl, Sebastian Padó

Abstract:

> We ask how to practically build a model for
> German named entity recognition (NER)
> that performs at the state of the art for both
> contemporary and historical texts,  i.e., a
> big-data and a small-data scenario. The two
> best-performing model families are pitted
> against each other (linear-chain CRFs and
> BiLSTM) to observe the trade-off between
> expressiveness and data requirements. BiL-
> STM  outperforms  the  CRF  when  large
> datasets are available and performs infe-
> rior  for  the  smallest  dataset. BiLSTMs
> profit substantially from transfer learning,
> which enables them to be trained on multi-
> ple corpora, resulting in a new state-of-the-
> art model for German NER on two contem-
> porary German corpora (CoNLL 2003 and
> GermEval 2014) and two historic corpora.

Material(s): [[Paper](http://aclweb.org/anthology/P18-2020.pdf)]

Comments: Many thanks to Martin Riedl for providing the datasets used
          in that paper 👍

## Recurrently Controlled Recurrent Networks

Authors: Yi Tay, Luu Anh Tuan, Siu Cheung Hui

Abstract:

> Recurrent neural networks (RNNs) such as long short-term memory and
> gated recurrent units are pivotal building blocks across a broad
> spectrum of sequence modeling problems. This paper proposes a
> recurrently controlled recurrent network (RCRN) for expressive and
> powerful sequence encoding. More concretely, the key idea behind our
> approach is to learn the recurrent gating functions using recurrent
> networks. Our architecture is split into two components - a controller
> cell and a listener cell whereby the recurrent controller actively
> influences the compositionality of the listener cell. We conduct
> extensive experiments on a myriad of tasks in the NLP domain such as
> sentiment analysis (SST, IMDb, Amazon reviews, etc.), question
> classification (TREC), entailment classification (SNLI, SciTail),
> answer selection (WikiQA, TrecQA) and reading comprehension
> (NarrativeQA). Across all 26 datasets, our results demonstrate that
> RCRN not only consistently outperforms BiLSTMs but also stacked
> BiLSTMs, suggesting that our controller architecture might be a
> suitable replacement for the widely adopted stacked architecture.

Material(s): [[Paper](https://arxiv.org/abs/1811.09786)]

## DARCCC: Detecting Adversaries by Reconstruction from Class Conditional Capsules

Authors: Nicholas Frosst, Sara Sabour, Geoffrey Hinton

Abstract:

> We present a simple technique that allows capsule models to detect adversarial
> images. In addition to being trained to classify images, the capsule model is
> trained to reconstruct the images from the pose parameters and identity of the
> correct top-level capsule. Adversarial images do not look like a typical member
> of the predicted class and they have much larger reconstruction errors when the
> reconstruction is produced from the top-level capsule for that class. We show
> that setting a threshold on the l2 distance between the input image and its
> reconstruction from the winning capsule is very effective at detecting
> adversarial images for three different datasets. The same technique works quite
> well for CNNs that have been trained to reconstruct the image from all or part
> of the last hidden layer before the softmax. We then explore a stronger,
> white-box attack that takes the reconstruction error into account. This attack
> is able to fool our detection technique but in order to make the model change
> its prediction to another class, the attack must typically make the
> "adversarial" image resemble images of the other class.

Material(s): [[Paper](https://arxiv.org/abs/1811.06969)] - [[Twitter](https://twitter.com/nickfrosst/status/1064593651026792448)]

## What do you learn from context? Probing for sentence structure in contextualized word representations

Authors: Anonymous

Abstract:

> Contextualized representation models such as CoVe (McCann et al., 2017)
> and ELMo (Peters et al., 2018a) have recently achieved state-of-the-art
> results on a diverse array of downstream NLP tasks. Building on recent
> token-level probing work, we introduce a novel edge probing task design
> and construct a broad suite of sub-sentence tasks derived from the
> traditional structured NLP pipeline. We probe word-level contextual
> representations from three recent models and investigate how they
> encode sentence structure across a range of syntactic, semantic, local,
> and long-range phenomena. We find that ELMo encodes linguistic
> structure at the word level better than other comparable models, and
> that existing models trained on language modeling and translation
> produce strong representations for syntactic phenomena, but only offer
> small improvements on semantic tasks over a noncontextual baseline.

Material(s): [[Paper](https://openreview.net/forum?id=SJzSgnRcKX)]

Comments: Under review as a conference paper at ICLR 2019

## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

Abstract:

> We introduce a new language representation model called BERT, which stands for
> Bidirectional Encoder Representations from Transformers. Unlike recent
> language representation models, BERT is designed to pre-train deep
> bidirectional representations by jointly conditioning on both left and right
> context in all layers. As a result, the pre-trained BERT representations can
> be fine-tuned with just one additional output layer to create state-of-the-art
> models for a wide range of tasks, such as question answering and language
> inference, without substantial task-specific architecture modifications.
> BERT is conceptually simple and empirically powerful. It obtains new
> state-of-the-art results on eleven natural language processing tasks,
> including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement),
> MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1
> question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming
> human performance by 2.0%.

Material(s): [[Paper](https://arxiv.org/abs/1810.04805)] - [[GitHub](https://github.com/google-research/bert)]

Comments: New deep bidirectional architecture for language understanding.
          Computationally ridiculous, *BERT_large* uses 256 TPU-days?!

## A Large-Scale Test Set for the Evaluation of Context-Aware Pronoun Translation in Neural Machine Translation

Authors: Mathias Müller, Annette Rios, Elena Voita, Rico Sennrich

Abstract:

> The translation of pronouns presents a special challenge to machine
> translation to this day, since it often requires context outside the
> current sentence. Recent work on models that have access to information
> across sentence boundaries has seen only moderate improvements in terms
> of automatic evaluation metrics such as BLEU. However, metrics that
> quantify the overall translation quality are ill-equipped to measure
> gains from additional context. We argue that a different kind of
> evaluation is needed to assess how well models translate
> inter-sentential phenomena such as pronouns. This paper therefore
> presents a test suite of contrastive translations focused specifically
> on the translation of pronouns. Furthermore, we perform experiments
> with several context-aware models. We show that, while gains in BLEU
> are moderate for those systems, they outperform baselines by a large
> margin in terms of accuracy on our contrastive test set. Our
> experiments also show the effectiveness of parameter tying for
> multi-encoder architectures.

Material(s): [[Paper](https://arxiv.org/abs/1810.02268)] - [[GitHub](https://github.com/ZurichNLP/ContraPro)]

## FRAGE: Frequency-Agnostic Word Representation

Authors: Chengyue Gong, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-Yan Liu

Abstract:

> Continuous word representation (aka word embedding) is a basic
> building block in many neural network-based models used in natural
> language processing tasks. Although it is widely accepted that words
> with similar semantics should be close to each other in the embedding
> space, we find that word embeddings learned in several tasks are biased
> towards word frequency: the embeddings of high-frequency and
> low-frequency words lie in different subregions of the embedding space,
> and the embedding of a rare word and a popular word can be far from
> each other even if they are semantically similar. This makes learned
> word embeddings ineffective, especially for rare words, and
> consequently limits the performance of these neural network models. In
> this paper, we develop a neat, simple yet effective way to learn
> *FRequency-AGnostic word Embedding* (FRAGE) using adversarial
> training. We conducted comprehensive studies on ten datasets across
> four natural language processing tasks, including word similarity,
> language modeling, machine translation and text classification. Results
> show that with FRAGE, we achieve higher performance than the baselines
> in all tasks.

Material(s): [[Paper](https://arxiv.org/abs/1809.06858)] - [[GitHub](https://github.com/ChengyueGongR/Frequency-Agnostic)]

## Pervasive Attention: 2D Convolutional Neural Networks for Sequence-to-Sequence Prediction

Authors: Maha Elbayad, Laurent Besacier, Jakob Verbeek

Abstract:

> Current state-of-the-art machine translation systems are based on
> encoder-decoder architectures, that first encode the input sequence, and then
> generate an output sequence based on the input encoding. Both are interfaced
> with an attention mechanism that recombines a fixed encoding of the source
> tokens based on the decoder state. We propose an alternative approach which
> instead relies on a single 2D convolutional neural network across both
> sequences. Each layer of our network re-codes source tokens on the basis of the
> output sequence produced so far. Attention-like properties are therefore
> pervasive throughout the network. Our model yields excellent results,
> outperforming state-of-the-art encoder-decoder systems, while being
> conceptually simpler and having fewer parameters.

Material(s): [[Paper](https://arxiv.org/abs/1808.03867)] - [[GitHub](https://github.com/elbayadm/attn2d)]

## Character-Level Language Modeling with Deeper Self-Attention

Authors: Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, Llion Jones

Abstract:

> LSTMs and other RNN variants have shown strong performance on
> character-level language modeling. These models are typically trained
> using truncated backpropagation through time, and it is common to
> assume that their success stems from their ability to remember
> long-term contexts. In this paper, we show that a deep (64-layer)
> transformer model with fixed context outperforms RNN variants by a
> large margin, achieving state of the art on two popular benchmarks-
> 1.13 bits per character on text8 and 1.06 on enwik8. To get good
> results at this depth, we show that it is important to add auxiliary
> losses, both at intermediate network layers and intermediate sequence
> positions.

Material(s): [[arxiv](https://arxiv.org/abs/1808.04444)]

## Contextual String Embeddings for Sequence Labeling

Authors: Alan Akbik, Duncan Blythe, Roland Vollgraf

Abstract:

> Recent advances in language modeling using recurrent neural networks
> have made it viable to model language as distributions over characters.
> By learning to predict the next character on the basis of previous
> characters, such models have been shown to automatically internalize
> linguistic concepts such as words, sentences, subclauses and even
> sentiment. In this paper, we propose to leverage the internal states of
> a trained character language model to produce a novel type of word
> embedding which we refer to as contextual string embeddings. Our
> proposed embeddings have the distinct properties that they (a) are
> trained without any explicit notion of words and thus fundamentally
> model words as sequences of characters, and (b) are contextualized by
> their surrounding text, meaning that the same word will have different
> embeddings depending on its contextual use. We conduct a comparative
> evaluation against previous embeddings and find that our embeddings are
> highly useful for downstream tasks: across four classic sequence
> labeling tasks we consistently outperform the previous
> state-of-the-art. In particular, we significantly outperform previous
> work on English and German named entity recognition (NER), allowing us
> to report new state-of-the-art F1-scores on the CONLL03 shared task.

Material(s): [[Paper](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view)] - [[GitHub](https://github.com/zalandoresearch/flair)]

Comments: New SOTAs in NER, POS and chunking.
          Clean and awesome implementation is available.
          Well written and detailed documentation for reproducing the
          results *is* also available. Accepted at COLING 2018.

## Neural Factor Graph Models for Cross-lingual Morphological Tagging

Authors: Chaitanya Malaviya, Matthew R. Gormley, Graham Neubig

Abstract:

> Morphological analysis involves predicting the syntactic traits of a word (e.g.
> {POS: Noun, Case: Acc, Gender: Fem}). Previous work in morphological tagging
> improves performance for low-resource languages (LRLs) through cross-lingual
> training with a high-resource language (HRL) from the same family, but is
> limited by the strict, often false, assumption that tag sets exactly overlap
> between the HRL and LRL. In this paper we propose a method for cross-lingual
> morphological tagging that aims to improve information sharing between
> languages by relaxing this assumption. The proposed model uses factorial
> conditional random fields with neural network potentials, making it possible to
> (1) utilize the expressive power of neural network representations to smooth
> over superficial differences in the surface forms, (2) model pairwise and
> transitive relationships between tags, and (3) accurately generate tag sets
> that are unseen or rare in the training data. Experiments on four languages
> from the Universal Dependencies Treebank demonstrate superior tagging
> accuracies over existing cross-lingual approaches.

Material(s): [[arxiv](https://arxiv.org/abs/1805.04570)] - [[GitHub](https://github.com/chaitanyamalaviya/NeuralFactorGraph)]

## Joint Embedding of Words and Labels for Text Classification

Authors: Guoyin Wang, Chunyuan Li, Wenlin Wang, Yizhe Zhang, Dinghan Shen,
         Xinyuan Zhang, Ricardo Henao, Lawrence Carin

Abstract:

> Word embeddings are effective intermediate representations for capturing
> semantic regularities between words, when learning the representations of text
> sequences. We propose to view text classification as a label-word joint
> embedding problem: each label is embedded in the same space with the word
> vectors. We introduce an attention framework that measures the compatibility of
> embeddings between text sequences and labels. The attention is learned on a
> training set of labeled samples to ensure that, given a text sequence, the
> relevant words are weighted higher than the irrelevant ones. Our method
> maintains the interpretability of word embeddings, and enjoys a built-in
> ability to leverage alternative sources of information, in addition to input
> text sequences. Extensive results on the several large text datasets show that
> the proposed framework outperforms the state-of-the-art methods by a large
> margin, in terms of both accuracy and speed.

Material(s): [[arxiv](https://arxiv.org/abs/1805.04174)] - [[GitHub](https://github.com/guoyinwang/LEAM)]

## Stack-Pointer Networks for Dependency Parsing

Authors: Xuezhe Ma, Zecong Hu, Jingzhou Liu, Nanyun Peng, Graham Neubig, Eduard Hovy

Abstract:

> We introduce a novel architecture for dependency parsing: *stack-pointer*
> networks (*StackPtr*). Combining pointer
> networks with an internal stack, the proposed model
> first reads and encodes the whole sentence, then builds the dependency tree
> top-down (from root-to-leaf) in a depth-first fashion. The stack tracks the
> status of the depth-first search and the pointer networks select one child for
> the word at the top of the stack at each step. The *StackPtr* parser
> benefits from the information of the whole sentence and all previously derived
> subtree structures, and removes the left-to-right restriction in classical
> transition-based parsers. Yet, the number of steps for building any (including
> non-projective) parse tree is linear in the length of the sentence just as
> other transition-based parsers, yielding an efficient decoding algorithm with
> O(n2) time complexity. We evaluate our model on 29 treebanks spanning 20
> languages and different dependency annotation schemas, and achieve
> state-of-the-art performance on 21 of them.

Material(s): [[arxiv](https://arxiv.org/abs/1805.01087)] - [[GitHub](https://github.com/XuezheMax/NeuroNLP2)]

## Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates

Authors: Taku Kudo

Abstract:

> Subword units are an effective way to alleviate the open vocabulary problems in
> neural machine translation (NMT). While sentences are usually converted into
> unique subword sequences, subword segmentation is potentially ambiguous and
> multiple segmentations are possible even with the same vocabulary. The question
> addressed in this paper is whether it is possible to harness the segmentation
> ambiguity as a noise to improve the robustness of NMT. We present a simple
> regularization method, subword regularization, which trains the model with
> multiple subword segmentations probabilistically sampled during training. In
> addition, for better subword sampling, we propose a new subword segmentation
> algorithm based on a unigram language model. We experiment with multiple
> corpora and report consistent improvements especially on low resource and
> out-of-domain settings.

Material(s): [[arxiv](https://arxiv.org/abs/1804.10959)] - [[GitHub](https://github.com/google/sentencepiece)]

## Unsupervised Neural Machine Translation with Weight Sharing

Authors: Zhen Yang, Wei Chen, Feng Wang, Bo Xu

Abstract:

> Unsupervised neural machine translation (NMT) is a recently proposed approach
> for machine translation which aims to train the model without using any labeled
> data. The models proposed for unsupervised NMT often use only one shared
> encoder to map the pairs of sentences from different languages to a
> shared-latent space, which is weak in keeping the unique and internal
> characteristics of each language, such as the style, terminology, and sentence
> structure. To address this issue, we introduce an extension by utilizing two
> independent encoders but sharing some partial weights which are responsible for
> extracting high-level representations of the input sentences. Besides, two
> different generative adversarial networks (GANs), namely the local GAN and
> global GAN, are proposed to enhance the cross-language translation. With this
> new approach, we achieve significant improvements on English-German,
> English-French and Chinese-to-English translation tasks.

Material(s): [[arxiv](https://arxiv.org/abs/1804.09057v1)] - [[GitHub](https://github.com/ZhenYangIACAS/unsupervised-NMT)]

## Exploiting Semantics in Neural Machine Translation with Graph Convolutional Networks

Authors: Yichao Lu, Phillip Keung, Faisal Ladhak, Vikas Bhardwaj, Shaonan Zhang, Jason Sun

Abstract:

> Semantic representations have long been argued as potentially useful for
> enforcing meaning preservation and improving generalization performance of
> machine translation methods. In this work, we are the first to incorporate
> information about predicate-argument structure of source sentences (namely,
> semantic-role representations) into neural machine translation. We use Graph
> Convolutional Networks (GCNs) to inject a semantic bias into sentence encoders
> and achieve improvements in BLEU scores over the linguistic-agnostic and
> syntax-aware versions on the English--German language pair.

Material(s): [[arxiv](https://arxiv.org/abs/1804.08313)]

## Adversarial Example Generation with Syntactically Controlled Paraphrase Networks

Authors: Mohit Iyyer, John Wieting, Kevin Gimpel, Luke Zettlemoyer

Abstract:

> We propose syntactically controlled paraphrase networks (SCPNs) and use them to
> generate adversarial examples. Given a sentence and a target syntactic form
> (e.g., a constituency parse), SCPNs are trained to produce a paraphrase of the
> sentence with the desired syntax. We show it is possible to create training
> data for this task by first doing backtranslation at a very large scale, and
> then using a parser to label the syntactic transformations that naturally occur
> during this process. Such data allows us to train a neural encoder-decoder
> model with extra inputs to specify the target syntax. A combination of
> automated and human evaluations show that SCPNs generate paraphrases that
> follow their target specifications without decreasing paraphrase quality when
> compared to baseline (uncontrolled) paraphrase systems. Furthermore, they are
> more capable of generating syntactically adversarial examples that both (1)
> "fool" pretrained models and (2) improve the robustness of these models to
> syntactic variation when used to augment their training data.

Material(s): [[arxiv](https://arxiv.org/abs/1804.06059)] - [[GitHub (coming soon)](https://github.com/miyyer/scpn)]

## Phrase-Based & Neural Unsupervised Machine Translation

Authors: Guillaume Lample, Myle Ott, Alexis Conneau, Ludovic Denoyer, Marc'Aurelio Ranzato

Abstract:

> Machine translation systems achieve near human-level performance on some
> languages, yet their effectiveness strongly relies on the availability of large
> amounts of bitexts, which hinders their applicability to the majority of
> language pairs. This work investigates how to learn to translate when having
> access to only large monolingual corpora in each language. We propose two model
> variants, a neural and a phrase-based model. Both versions leverage automatic
> generation of parallel data by backtranslating with a backward model operating
> in the other direction, and the denoising effect of a language model trained on
> the target side. These models are significantly better than methods from the
> literature, while being simpler and having fewer hyper-parameters. On the
> widely used WMT14 English-French and WMT16 German-English benchmarks, our
> models respectively obtain 27.1 and 23.6 BLEU points without using a single
> parallel sentence, outperforming the state of the art by more than 11 BLEU
> points.

Material(s): [[arxiv](https://arxiv.org/abs/1804.07755)]

Comments: Nice new SOTA results, but **unfortunately** no implementation is
          currently provided, so the results are **not** reproducible.

## GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding

Authors: Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman

Abstract:

> For  natural  language  understanding  (NLU) technology to be maximally useful,
> both practically  and  as  a  scientific  object  of  study,  it must be
> general: it must be able to process language in a way that is not exclusively
> tailored to any one specific task or dataset. In pursuit of this objective, we
> introduce the General Lan- guage  Understanding  Evaluation  benchmark (GLUE),
> a  tool  for  evaluating  and  analyzing the  performance  of  models  across
> a  diverse range of existing NLU tasks. GLUE is model- agnostic, but it
> incentivizes sharing knowledge across  tasks  because  certain  tasks  have
> very limited  training  data.    We  further  provide  a hand-crafted
> diagnostic test suite that enables detailed  linguistic  analysis  of  NLU
> models. We evaluate baselines based on current meth- ods  for  multi-task  and
> transfer  learning  and find that they do not immediately give substantial
> improvements  over  the  aggregate  performance of training a separate model
> per task, indicating room for improvement in develop- ing general and robust
> NLU systems.

Material(s): [[Paper](https://www.nyu.edu/projects/bowman/glue.pdf)] - [[Website](https://gluebenchmark.com/)]

## The unreasonable effectiveness of the forget gate

Authors: Jos van der Westhuizen, Joan Lasenby

Abstract:

> Given the success of the gated recurrent unit, a natural question is whether
> all the gates of the long short-term memory (LSTM) network are necessary.
> Previous research has shown that the forget gate is one of the most important
> gates in the LSTM. Here we show that a forget-gate-only version of the LSTM
> with chrono-initialized biases, not only provides computational savings but
> outperforms the standard LSTM on multiple benchmark datasets and competes with
> some of the best contemporary models. Our proposed network, the JANET, achieves
> accuracies of 99% and 92.5% on the MNIST and pMNIST datasets, outperforming the
> standard LSTM which yields accuracies of 98.5% and 91%.

Material(s): [[arxiv](https://arxiv.org/abs/1804.04849)] - [[GitHub](https://github.com/JosvanderWesthuizen/janet)]

## When and Why are Pre-trained Word Embeddings Useful for Neural Machine Translation?

Authors: Ye Qi, Devendra Singh Sachan, Matthieu Felix, Sarguna Janani Padmanabhan, Graham Neubig

Abstract:

> The performance of Neural Machine Translation (NMT) systems often suffers in
> low-resource scenarios where sufficiently large-scale parallel corpora cannot
> be obtained. Pre-trained word embeddings have proven to be invaluable for
> improving performance in natural language analysis tasks, which often suffer
> from paucity of data. However, their utility for NMT has not been extensively
> explored. In this work, we perform five sets of experiments that analyze when
> we can expect pre-trained word embeddings to help in NMT tasks. We show that
> such embeddings can be surprisingly effective in some cases -- providing gains
> of up to 20 BLEU points in the most favorable setting.

Material(s): [[arxiv](https://arxiv.org/abs/1804.06323)] - [[GitHub](https://github.com/neulab/word-embeddings-for-nmt)]

## Adafactor: Adaptive Learning Rates with Sublinear Memory Cost

Authors: Noam Shazeer, Mitchell Stern

Abstract:

> In several recently proposed stochastic optimization methods (e.g. RMSProp,
> Adam, Adadelta), parameter updates are scaled by the inverse square roots of
> exponential moving averages of squared past gradients. Maintaining these
> per-parameter second-moment estimators requires memory equal to the number of
> parameters. For the case of neural network weight matrices, we propose
> maintaining only the per-row and per-column sums of these moving averages, and
> estimating the per-parameter second moments based on these sums. We demonstrate
> empirically that this method produces similar results to the baseline.
> Secondly, we show that adaptive methods can produce larger-than-desired updates
> when the decay rate of the second moment accumulator is too slow. We propose
> update clipping and a gradually increasing decay rate scheme as remedies.
> Combining these methods and dropping momentum, we achieve comparable results to
> the published Adam regime in training the Transformer model on the WMT 2014
> English-German machine translation task, while using very little auxiliary
> storage in the optimizer. Finally, we propose scaling the parameter updates
> based on the scale of the parameters themselves.

Material(s): [[arxiv](https://arxiv.org/abs/1804.04235)]

## Fine-Grained Attention Mechanism for Neural Machine Translation

Authors: Heeyoul Choi, Kyunghyun Cho, Yoshua Bengio

Abstract:

> Neural machine translation (NMT) has been a new paradigm in machine
> translation, and the attention mechanism has become the dominant approach with
> the state-of-the-art records in many language pairs. While there are variants
> of the attention mechanism, all of them use only temporal attention where one
> scalar value is assigned to one context vector corresponding to a source word.
> In this paper, we propose a fine-grained (or 2D) attention mechanism where each
> dimension of a context vector will receive a separate attention score. In
> experiments with the task of En-De and En-Fi translation, the fine-grained
> attention method improves the translation quality in terms of BLEU score. In
> addition, our alignment analysis reveals how the fine-grained attention
> mechanism exploits the internal structure of context vectors.

Material(s): [[arxiv](https://arxiv.org/abs/1803.11407)]

Comments: References are broken. Baseline is not really strong. The resulting
BLEU-score of 23.74 for English-German is much lower than [Luong and
Manning](https://arxiv.org/abs/1508.04025) reported (25.9) in 2015!

## Investigating Capsule Networks with Dynamic Routing for Text Classification

Authors: Wei Zhao, Jianbo Ye, Min Yang, Zeyang Lei, Suofei Zhang, Zhou Zhao

Abstract:

> In this study, we explore capsule networks with dynamic routing for text
> classification. We propose three strategies to stabilize the dynamic routing
> process to alleviate the disturbance of some noise capsules which may contain
> "background" information or have not been successfully trained. A series of
> experiments are conducted with capsule networks on six text classification
> benchmarks. Capsule networks achieve state of the art on 4 out of 6 datasets,
> which shows the effectiveness of capsule networks for text classification. We
> additionally show that capsule networks exhibit significant improvement when
> transfer single-label to multi-label text classification over strong baseline
> methods. To the best of our knowledge, this is the first work that capsule
> networks have been empirically investigated for text modeling.

Material(s): [[arxiv](https://arxiv.org/abs/1804.00538)]

## Mittens: An Extension of GloVe for Learning Domain-Specialized Representations

Authors: Nicholas Dingwall, Christopher Potts

Abstract:

> We present a simple extension of the GloVe representation learning model that
> begins with general-purpose representations and updates them based on data from
> a specialized domain. We show that the resulting representations can lead to
> faster learning and better results on a variety of tasks.

Material(s): [[arxiv](https://arxiv.org/abs/1803.09901)] - [[GitHub](https://github.com/roamanalytics/mittens)]

## An Analysis of Neural Language Modeling at Multiple Scales

Authors: Stephen Merity, Nitish Shirish Keskar, Richard Socher

Abstract:

> Many of the leading approaches in language modeling introduce novel, complex
> and specialized architectures. We take existing state-of-the-art word level
> language models based on LSTMs and QRNNs and extend them to both larger
> vocabularies as well as character-level granularity. When properly tuned, LSTMs
> and QRNNs achieve state-of-the-art results on character-level (Penn Treebank,
> enwik8) and word-level (WikiText-103) datasets, respectively. Results are
> obtained in only 12 hours (WikiText-103) to 2 days (enwik8) using a single
> modern GPU.

Material(s): [[arxiv](https://arxiv.org/abs/1803.08240)]

## Word2Bits - Quantized Word Vectors

Author: Maximilian Lam

Abstract:

> Word vectors require significant amounts of memory and storage, posing issues
> to resource limited devices like mobile phones and GPUs. We show that high
> quality quantized word vectors using 1-2 bits per parameter can be learned by
> introducing a quantization function into Word2Vec. We furthermore show that
> training with the quantization function acts as a regularizer. We train word
> vectors on English Wikipedia (2017) and evaluate them on standard word
> similarity and analogy tasks and on question answering (SQuAD). Our quantized
> word vectors not only take 8-16x less space than full precision (32 bit) word
> vectors but also outperform them on word similarity tasks and question
> answering.

Material(s): [[arxiv](https://arxiv.org/abs/1803.05651)] - [[GitHub](https://github.com/agnusmaximus/Word2Bits)]

## FEVER: a large-scale dataset for Fact Extraction and VERification

Authors: James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal

Abstract:

> Unlike other tasks and despite recent interest, research in textual claim
> verification has been hindered by the lack of large-scale manually annotated
> datasets. In this paper we introduce a new publicly available dataset for
> verification against textual sources, FEVER: Fact Extraction and VERification.
> It consists of 185,441 claims generated by altering sentences extracted from
> Wikipedia and subsequently verified without knowledge of the sentence they were
> derived from. The claims are classified as Supported, Refuted or NotEnoughInfo
> by annotators achieving 0.6841 in Fleiss κ. For the first two classes, the
> annotators also recorded the sentence(s) forming the necessary evidence for
> their judgment. To characterize the challenge of the dataset presented, we
> develop a pipeline approach using both baseline and state-of-the-art components
> and compare it to suitably designed oracles. The best accuracy we achieve on
> labeling a claim accompanied by the correct evidence is 31.87%, while if we
> ignore the evidence we achieve 50.91%. Thus we believe that FEVER is a
> challenging testbed that will help stimulate progress on claim verification
> against textual sources.

Material(s): [[arxiv](https://arxiv.org/abs/1803.05355)] - [[Website](https://sheffieldnlp.github.io/fever/)]

Comments: Really interesting, new problem. I'm going to take part in that
          competition.

## Learning to Explain: An Information-Theoretic Perspective on Model Interpretation

Authors: Jianbo Chen, Le Song, Martin J. Wainwright, Michael I. Jordan

Abstract:

> We introduce instancewise feature selection as a methodology for model
> interpretation. Our method is based on learning a function to extract a subset
> of features that are most informative for each given example. This feature
> selector is trained to maximize the mutual information between selected
> features and the response variable, where the conditional distribution of the
> response variable given the input is the model to be explained. We develop an
> efficient variational approximation to the mutual information, and show that
> the resulting method compares favorably to other model explanation methods on a
> variety of synthetic and real data sets using both quantitative metrics and
> human evaluation.

Material(s): [[arxiv](https://arxiv.org/abs/1802.07814)] - [[GitHub](https://github.com/Jianbo-Lab/L2X)]

## CoVeR: Learning Covariate-Specific Vector Representations with Tensor Decompositions

Authors: Kevin Tian, Teng Zhang, James Zou

Abstract:

> Word embedding is a useful approach to capture co-occurrence structures in a
> large corpus of text. In addition to the text data itself, we often have
> additional covariates associated with individual documents in the corpus---e.g.
> the demographic of the author, time and venue of publication, etc.---and we
> would like the embedding to naturally capture the information of the
> covariates. In this paper, we propose CoVeR, a new tensor decomposition model
> for vector embeddings with covariates. CoVeR jointly learns a base embedding
> for all the words as well as a weighted diagonal transformation to model how
> each covariate modifies the base embedding. To obtain the specific embedding
> for a particular author or venue, for example, we can then simply multiply the
> base embedding by the transformation matrix associated with that time or venue.
> The main advantages of our approach is data efficiency and interpretability of
> the covariate transformation matrix. Our experiments demonstrate that our joint
> model learns substantially better embeddings conditioned on each covariate
> compared to the standard approach of learning a separate embedding for each
> covariate using only the relevant subset of data, as well as other related
> methods. Furthermore, CoVeR encourages the embeddings to be "topic-aligned" in
> the sense that the dimensions have specific independent meanings. This allows
> our covariate-specific embeddings to be compared by topic, enabling downstream
> differential analysis. We empirically evaluate the benefits of our algorithm on
> several datasets, and demonstrate how it can be used to address many natural
> questions about the effects of covariates.

Material(s): [[arxiv](https://arxiv.org/abs/1802.07839)]

## Fooling OCR Systems with Adversarial Text Images

Authors: Congzheng Song, Vitaly Shmatikov

Abstract:

> We demonstrate that state-of-the-art optical character recognition (OCR) based
> on deep learning is vulnerable to adversarial images. Minor modifications to
> images of printed text, which do not change the meaning of the text to a human
> reader, cause the OCR system to "recognize" a different text where certain
> words chosen by the adversary are replaced by their semantic opposites. This
> completely changes the meaning of the output produced by the OCR system and by
> the NLP applications that use OCR for preprocessing their inputs.

Material(s): [[arxiv](https://arxiv.org/abs/1802.05385)]

## TextZoo, a New Benchmark for Reconsidering Text Classification

Authors: Benyou Wang, Li Wang, Qikang Wei

Abstract:

> Text representation is a fundamental concern in Natural Language Processing,
> especially in text classification. Recently, many neural network approaches
> with delicate representation model (e.g. FASTTEXT, CNN, RNN and many hybrid
> models with attention mechanisms) claimed that they achieved state-of-art in
> specific text classification datasets. However, it lacks an unified benchmark
> to compare these models and reveals the advantage of each sub-components for
> various settings. We re-implement more than 20 popular text representation
> models for classification in more than 10 datasets. In this paper, we
> reconsider the text classification task in the perspective of neural network
> and get serval effects with analysis of the above results.

Material(s): [[arxiv](https://arxiv.org/abs/1802.03656)] - [[GitHub](https://github.com/wabyking/TextClassificationBenchmark)]

Comments: Early draft of the paper

## Fast and Accurate Reading Comprehension by Combining Self-Attention and Convolution

Authors: Adams Wei Yu, David Dohan, Quoc Le, Thang Luong, Rui Zhao, Kai Chen

Abstract:

> Current end-to-end machine reading and question answering (Q\&A) models are
> primarily based on recurrent neural networks (RNNs) with attention. Despite
> their success, these models are often slow for both training and inference due
> to the sequential nature of RNNs. We propose a new Q\&A model that does not
> require recurrent networks:  It consists exclusively of attention and
> convolutions, yet achieves equivalent or better performance than existing
> models. On the SQuAD dataset, our model is 3x to 13x faster in training and 4x
> to 9x faster in inference. The speed-up gain allows us to train the model with
> much more data. We hence  combine our model with data generated by
> backtranslation from a neural machine translation model. This data augmentation
> technique  not only enhances the training examples but also diversifies the
> phrasing of the sentences, which results in immediate accuracy improvements.
> Our single model achieves 84.6 F1 score on the test set, which is significantly
> better than the best published F1 score of 81.8.

Material(s): [[ICLR 2018](https://openreview.net/forum?id=B14TlG-RW)]

Comments: Enormous reduction of training time. "Data augmentation" using
          neural machine translation is an awesome idea.

## Matrix capsules with EM routing

Authors: Geoffrey E Hinton, Sara Sabour, Nicholas Frosst

Abstract:

> A capsule is a group of neurons whose outputs represent different properties of
> the same entity. Each layer in a capsule network contains many capsules [a
> group of capsules forms a capsule layer and can be used in place of a
> traditional layer in a neural net]. We describe a version of capsules in which
> each capsule has a logistic unit to represent the presence of an entity and a
> 4x4 matrix which could learn to represent the relationship between that entity
> and the viewer (the pose). A capsule in one layer votes for the pose matrix of
> many different capsules in the layer above by multiplying its own pose matrix
> by trainable viewpoint-invariant transformation matrices that could learn to
> represent part-whole relationships. Each of these votes is weighted by an
> assignment coefficient. These coefficients are iteratively updated for each
> image using the Expectation-Maximization algorithm such that the output of each
> capsule is routed to a capsule in the layer above that receives a cluster of
> similar votes. The transformation matrices are trained discriminatively by
> backpropagating through the unrolled iterations of EM between each pair of
> adjacent capsule layers. On the smallNORB benchmark, capsules reduce the number
> of test errors by 45\% compared to the state-of-the-art. Capsules also show far
> more resistance to white box adversarial attack than our baseline convolutional
> neural network.

Material(s): [[ICLR 2018](https://openreview.net/forum?id=HJWLfGWRb)] - [[GitHub, offical implementation of CapsNet](https://github.com/tensorflow/models/pull/3265)]

Comments: Finally accepted!

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

Material(s): [[arxiv](https://arxiv.org/abs/1802.05365)] - [[Github, part of allennlp](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)] - [[Website](http://allennlp.org/elmo)]

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

Material(s): [[arxiv](https://arxiv.org/abs/1801.06146)] - [[GitHub, part of fastai courses](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb)]

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
