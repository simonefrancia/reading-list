# Reading list

This repository contains my reading list for deep learning research. Topics
are mainly: relation extraction, reading comprehension, neural network
architectures, classification, neural machine translation and many more.

# 2018

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

Materials(s): [[arxiv](https://arxiv.org/abs/1804.04235)]

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

Material(s): [[arxiv](https://arxiv.org/abs/1803.05651) - [GitHub](https://github.com/agnusmaximus/Word2Bits)]

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

Material(s): [[arxiv](https://arxiv.org/abs/1803.05355) - [Website](https://sheffieldnlp.github.io/fever/)]

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

Material(s): [[arxiv](https://arxiv.org/abs/1802.07814) - [GitHub](https://github.com/Jianbo-Lab/L2X)]

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

Material(s): [[arxiv](https://arxiv.org/abs/1802.03656) - [GitHub](https://github.com/wabyking/TextClassificationBenchmark)]

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
