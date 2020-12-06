## Graph Mining @ NeurIPS

The Graph Mining team at Google is excited to be presenting at the 2020 NeurIPS Conference. Please join us on Sunday, December 6th, at 1PM EST. The Expo information page can be found [here](https://nips.cc/Conferences/2020/Schedule?showEvent=20237). This page will be updated with video links after the workshop.

To read more about the Graph Mining team, check out our [research page](https://research.google/teams/algorithms-optimization/graph-mining/).

<p align="center">
  <img width="75" src="/graph-mining-logo-1.png">
</p>

## Introduction
_Vahab Mirrokni_  
[Research page](https://research.google/people/mirrokni/)

In this talk, Vahab Mirrokni, the founder of the Graph Mining team, introduces Graph Mining and Learning at a high level. This talk touches on what graphs are, why they are important, and where they appear in the world of big data. The talk then dives into the core tools that make up the Graph Mining and Learning toolbox, and lays out several canonical use cases. It also touches upon discussion of how to combine algorithms, systems, and machine learning to build a scalable graph-based learning system in different distributed environments. Finally, it provides a brief history of graph mining and learning projects at Google. The talk sets up the rest of the Expo by introducing common terms and themes that will appear in the following talks.

## Application Stories

#### Modeling COVID with Graph Neural Networks
_Amol Kapoor_  
[Research page](https://research.google/people/106318/)

In this talk, Amol Kapoor discusses how GNNs can be used to predict the change in COVID caseloads in US counties when supplemented with spatio-temporal mobility information. This presentation covers the recent paper: _Examining COVID-19 Forecasting using Spatio-Temporal Graph Neural Networks_ ([arxiv](https://arxiv.org/abs/2007.03113)). 

#### Using Graph Mining for Privacy 
_Alessandro Epasto_  
[Research page](https://research.google/people/AlessandroEpasto/)

In this talk, Alessandro Epasto reviews applications of graph mining to privacy. First, we will see an application of graph-based clustering to the privacy-preserving Google effort of Federated Learning of Cohort (FLOC). Second, we will discuss research on how to ensure privacy on graph-based applications using on-device computations. For more on FLOC, take a look at the [public announcement](https://github.com/google/ads-privacy/blob/master/proposals/FLoC/README.md). This presentation covers the following papers: _On-Device Algorithms for Public-Private Data with Absolute Privacy_ ([pdf](https://epasto.org/papers/www2019ondevice.pdf)); and _Efficient Algorithms for Private-Public Graphs_ ([pdf](https://epasto.org/papers/kdd2015.pdf)).

#### Causal Inference
_Jean Pouget-Abadie_  
[Research page](https://research.google/people/JeanPougetAbadie/)

In this short talk, we look at how clustering can be used to run better randomized experiments. Randomized experiments allow us to estimate causal effects, but such estimation suffers when the units of interest are not independent. Clustering is used to mitigate this problem by avoiding interactions between groups of units with different treatment assignments. We also take a specific look at market experiments where different clustering techniques may be more suitable. This talk covers the following papers: _Variance Reduction in Bipartite Experiments through Correlation Clustering_ ([pdf](https://papers.nips.cc/paper/2019/file/bc047286b224b7bfa73d4cb02de1238d-Paper.pdf)); and _Randomized Experimental Design via Geographic Clustering_ ([pdf](https://dl.acm.org/doi/pdf/10.1145/3292500.3330778)).

## Graph Mining At Scale

#### Grale: Learning Graphs
_Jonathan Halcrow_

In this talk, Jonathan Halcrow discusses the Grale graph building framework, a highly scalable tool for generating learned similarity graphs from arbitrary data. This talk covers the recent paper: _Grale: Designing Networks for Graph Learning_ ([arxiv](https://arxiv.org/abs/2007.12002?)). 

#### Similarity Ranking
_Alessandro Epasto_

In this talk, Alessandro Epasto addresses the following question: how can we measure the similarity of two nodes in a graph? Similarity rankings have important applications ranging from recommender systems, link prediction and anomaly detection. We will review standard techniques in unsupervised graph similarity ranking with a focus on scalable algorithms. We will also show some recent applications of similarity ranking. This presentation covers the following papers: _Ego-net Community Mining Applied to Friend Suggestion_ ([pdf](http://www.vldb.org/pvldb/vol9/p324-epasto.pdf)); and _Reduce and Aggregate: Similarity Rankings in Multi-Categorical Bipartite Graphs_ ([pdf](https://www.epasto.org/papers/reduce-aggregate.pdf)).

#### Clustering at Scale
_Vahab Mirrokni_

In this talk, Vahab Mirrokni provides an overview of clustering at scale. The talk starts with affinity hierarchical clustering ([pdf](http://papers.neurips.cc/paper/7262-affinity-clustering-hierarchical-clustering-at-scale.pdf)), which provides the backbone of other clustering algorithms. The talk then covers a scalable distributed balanced partitioning algorithm ([arxiv](https://arxiv.org/abs/1512.02727)), and highlights an application of balanced partitioning in cache-aware load balancing to save 32% flash bandwidth ([pdf](http://www.vldb.org/pvldb/vol12/p709-archer.pdf)). Finally, it discusses the techniques of distributed composable core-set and sketching and how they are applied to develop distributed algorithms for k-clustering and k-cover ([pdf](https://www.cs.utah.edu/~bhaskara/files/balanced-dist.pdf), [pdf](https://dl.acm.org/ft_gateway.cfm?id=3220081&type=pdf)).

#### Community Detection
_Jakub "Kuba" Łącki_  
[Research page](https://research.google/people/105517/)

In this talk, Jakub Łącki presents graph clustering techniques that can be used to find communities in a social network. In addition to reviewing some well-known techniques, the talk introduces a new method for detecting and evaluating communities. The new method exhibits competitive empirical performance and good theoretical properties.

#### Label Propagation
_Allan Heydon_  
[Research page](https://research.google/people/AllanHeydon/)

In this talk, Allan Heydon describes one of Google’s systems for doing large-scale semi-supervised learning via label propagation. The algorithm requires only a small fraction of the input data instances to be labeled, and works by iteratively propagating labels along the edges of a similarity graph. Because it is implemented as a massively parallel computation, it scales to graphs with XT edges, XXXB nodes, and potentially millions of distinct labels. Due to the generality of the data model, it can be applied to a wide variety of problems, including spam/abuse detection, image/video labeling, natural language processing, noisy label cleaning, and label augmentation for downstream supervised model training. More can be found on the [Google Research Blog](https://ai.googleblog.com/2016/10/graph-powered-machine-learning-at-google.html). This presentation covers the paper: _Large Scale Distributed Semi-Supervised Learning Using Streaming Approximation_ ([arxiv](https://arxiv.org/abs/1512.01752)).

## Graph Neural Networks

#### GNNs and Graph Embeddings
_Bryan Perozzi_  
[Research page](https://research.google/people/BryanPerozzi/)

In this talk, Bryan Perozzi presents an overview of Graph Embeddings and Graph Convolutions. The talk begins with a high level discussion of graph embeddings -- how they are created and why they are useful. The talk then shifts to talk about Graph Convolutions. It covers how graph embeddings are used in graph convolutions, and why graph convolutional networks provide a flexible and powerful way of incorporating node context in a single unifying deep ml framework. Finally, the talk closes out with a brief discussion of some of the challenges of graph learning. This talk covers the following papers: _DeepWalk: Online Learning of Social Representations_ ([pdf](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)); _Semi-Supervised Classification with Graph Convolutional Networks_ ([arxiv](https://arxiv.org/abs/1609.02907)); _Neural Message Passing for Quantum Chemistry_ ([arxiv](https://arxiv.org/abs/1704.01212)); _N-GCN: Multi-scale Graph Convolution for Semi-supervised Node Classification_ ([arxiv](https://arxiv.org/abs/1802.08888)); and _MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing_ ([arxiv](https://arxiv.org/abs/1905.00067)).

#### PPRGo: GNNs at Scale
_Amol Kapoor_

In this talk, Amol Kapoor talks about some of the challenges with running GNNs at scale, and presents a solution called PPRGo. This presentation covers the recent paper: _Scaling Graph Neural Networks with Approximate PageRank_ ([arxiv](https://arxiv.org/abs/2007.01570)). 

#### Debiasing GNNs
_John Palowitch_  
[Research page](https://research.google/people/JohnPalowitch/)

In this talk, John Palowitch talks about a training-time projection for debiasing graph representations learned from unsupervised algorithms. This presentation covers the recent paper: _Debiasing Graph Embeddings via the Metadata-Orthogonal Training Unit_ ([arxiv](https://arxiv.org/abs/1909.11793)).

#### Learning Multiple Embeddings
_Alessandro Epasto_

In this talk, Alessandro Epasto presents recent advances in learning graph embeddings. We will show a novel methodology to learning multiple embeddings per node that allows us to understand better the community structure of the graph and obtain improved results in downstream ML tasks such as link prediction. Our method is based on the Persona Graph method, a novel framework for graph analysis that identifies clusters in complex networks through the use of ego-network analysis. This presentation covers the following papers: _Is a Single Embedding Enough? Learning Node Representations that Capture Multiple Social Contexts_
 ([arxiv](https://arxiv.org/abs/1905.02138)); and _Ego-splitting Framework: from Non-Overlapping to Overlapping Clusters_ ([pdf](https://www.epasto.org/papers/kdd2017.pdf)).

## Algorithms, Systems, and Scalability

#### Graph Tensor
_Martin Blais_  
[Research page](https://research.google/people/MartinBlais/)

In this talk, Martin Blais discusses the infrastructure required to train graph learning models at Google scale.

#### Distributed Graph Algorithms
_Jakub "Kuba" Łącki_

In this talk, Jakub Łącki describes the challenges and techniques for processing trillion-edge graphs. The talk discusses how practical aspects of running a distributed computation are captured in the theoretical computation models, as well as how modelling and algorithmic advancements result in better empirical running times. This talk covers material from multiple research papers, including: _Connected Components at Scale via Local Contractions_ ([arxiv](https://arxiv.org/abs/1807.10727)), _Massively Parallel Computation via Remote Memory Access_ ([arxiv](https://arxiv.org/abs/1905.07533)), and _Parallel Graph Algorithms in Constant Adaptive Rounds: Theory meets Practice_ ([arxiv](https://arxiv.org/abs/2009.11552)).

#### Multi-core Parallel Graph Clustering
_Jakub "Kuba" Łącki_

In this talk, Jakub Łącki presents single-machine parallel clustering algorithms that can cluster graphs of billions of edges in a few minutes.

## Acknowledgements

The presentations above are based on the work of many folks in Graph Mining. Without their hard work and dedication, this expo could not happen. Special thanks to:

Aaron Archer,
André Linhares,
Andrew Tomkins,
Arjun Gopalan,
Ashkan Fard,
CJ Carey,
David Eisenstat, 
Dustin Zelle,
Filipe Almeida,
Hossein Esfandiari,
Kevin Aydin,
Jason Lee,
Matthew Fahrbach,
MohammadHossein Bateni,
Nikos Parotsidis,
Reah Miyara,
Sam Ruth,
Silvio Lattanzi,
Warren Schudy,
and many collaborators.
