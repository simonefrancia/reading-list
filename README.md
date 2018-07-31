# Reading list

This repository contains my reading list for deep learning research. Topics
are mainly: relation extraction, reading comprehension, neural network
architectures, classification, neural machine translation and many more.

# 2018

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
