---
layout: post
title:  Pay Attention, Relations are important
date:   2015-03-15 16:40:16
description: Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs
---

*This blog is a work in progress. Please check after few days. Thank you.*


<center><img align="middle" src="{{ site.baseurl }}/assets/img/hulkAttention.jpg"></center>
<!-- ![something something blah blah]({{ site.baseurl }}/assets/img/hulkAttention.jpg) -->

### Overview
In recent years, Knowledge Graphs have been utilized to solve many real world problems such as Semantic Search, Dialogue Generation and Question Answering to name a few. Knowledge Graphs suffer from incompleteness in form of missing entities and relations, which has fueled a lot of research on Knowledge Base completion. Relation prediction is one of the widely used approaches to solve the problem of incompleteness.  

Here we will present our ACL 2019 work on [Knowledge Base Attention Network](https://arxiv.org/abs/1906.01195), novel neural network architecture which operates on Knowledge Graphs and learns to extract more expressive feature representations for entities and relations. Our model also addresses the shortcomings of previous methods like ConvKB, ConvE, RGCN, TransE, DistMult and ComplEx.

### Graph Convolution Networks

Convolutional Neural Networks (CNNs) have helped in significantly improving the state-of-the-art in Computer Vision research. Image data can be seen as a *spatial grid* which is highly rigid (each pixel is connected to it's 8 neighboring pixels). The CNNs exploit the rigidity and regular connectivity pattern of image data and thus gives us an effective and trivial method to implement convolution operator.

Convolution operator in images gathers information from neighboring pixels commensurately. Similar idea is used when defining convolution operation on graphs. <br/>
Now Consider, a graph with $$ n $$ nodes, specified as set of node features $$ \textbf{x} = \{\vec{x}_{1},\vec{x}_{2},...,\vec{x}_{N}\} $$ and the connectivity information in form of adjacency matrix $$ A $$. A *graph convolutional layer* then produces a transformed set of node feature vectors $$ \textbf{x}^\prime = \{\vec{x}_{1}^{\prime},\vec{x}_{2}^{\prime},...,\vec{x}_{N}^{\prime}\} $$ based on the structure of the graph and intial embeddings of the entities.

The convolution operation on graph can be summarized with the help of following two operations. First, in order to achieve a higher order representation of nodes, we do a linear transformation parametrized by a weight matrix $$ \textbf{W} $$. The transformed feature vectors $$ \vec{g_{i}} $$ are given as $$\vec{g_{i}} = \textbf{W}\vec{x_{i}} $$. Finally, to get the output features of node $$ i $$, we will aggregate the features across the neighborhood of the node. Final feature vectors $$ \vec{x^{'}_{i}} $$ can be defined as:

$$
\begin{aligned}
\vec{x_{i}^{\prime}} = \sigma \Bigg( \sum_{j \in \mathcal{N}_{i}} \alpha_{ij} \vec{g_{j}} \Bigg)
\end{aligned}
$$

where $$ \sigma $$ is an activation function, and $$ \alpha_{ij} $$ specifies the importance of node $$ j^{'}s $$ features to node $$ i $$. <br/>
In most of the prior works, $$ \alpha_{ij} $$ is defined explicitly, based on structural properties or as a learnable weight.

A single GAT layer can be described as
\begin{equation}\label{eq:eij}
 e_{ij} = a(\textbf{W} \vec{x_{i}},\textbf{W} \vec{x_{j}})
\end{equation}
where \(e_{ij}\) is the attention value of the edge \((e_i, e_j)\) in $\mathcal{G}$, \(\textbf{W}\) is a parametrized linear transformation matrix mapping the input features to a higher dimensional output feature space, and \(a\) is any \emph{attention function} of our choosing.
%where \(a\) is an attention mechanism of choice, GAT uses a single layer feed-forward neural network, \(e_{ij}\) is the attention value of the edge \((e_i, e_j)\) in the graph and \(\textbf{W}\) is a linear transformation matrix used to map the features from input space to output space.

Attention values for each edge are the \textit{importance} of the edge \((e_i, e_j)^{'}\)s features for a source node \(e_i\). Here, the relative attention \(\alpha_{ij}\) is computed using a \emph{softmax function} over all the values in the neighborhood.
% as given in equation \ref{eq:alphaij}.
%\begin{equation}\label{eq:alphaij}
%\alpha_{ij} = \textrm{softmax}_{j} (e_{ij})
%\end{equation}
Equation \ref{eq:hiprime} shows the output of a layer. GAT employs \emph{multi-head attention} to stabilize the learning process as credited to \cite{NIPS2017_7181}.
\begin{equation}\label{eq:hiprime}
\vec{x_{i}^{\prime}} = \sigma \Bigg( \sum_{j \in \mathcal{N}_{i}} \alpha_{ij} \textbf{W} \vec{x_{j}} \Bigg)
\end{equation}
The multihead attention process of concatenating $K$ attention heads is shown as follows in Equation \ref{eq:hiconcat}.
\begin{equation}\label{eq:hiconcat}
 \vec{x_{i}^{\prime}} = \underset{k=1}{\stackrel{K}{\Big \Vert}}  \sigma \Bigg( \sum_{j \in \mathcal{N}_{i}} \alpha_{ij}^{k} \textbf{W}^{k} \vec{x_{j}} \Bigg)
\end{equation}
where $\Vert$ represents concatenation, \(\sigma\) represents any non-linear function, $\alpha_{ij}^{k}$ are normalized attention coefficients of edge \((e_i, e_j)\) calculated by the $k$-th attention mechanism, and \(\textbf{W}^k\) represents the corresponding linear transformation matrix of the \(k\)-th
attention mechanism.
The output embedding in the final layer is calculated using \emph{averaging}, instead of the concatenation operation, to achieve multi-head attention, as is shown in the following Equation \ref{eq:hisummation}.
\begin{equation}\label{eq:hisummation}
 \vec{x_{i}^{\prime}} =   \sigma \Bigg(\frac{1}{K}\sum_{k = 1}^{K}\sum_{j \in \mathcal{N}_{i}} \alpha_{ij}^{k} \textbf{W}^{k} \vec{x_{j}} \Bigg)
\end{equation}

### Attention Networks

### KBAT (Knowledge Base Attention) Network

Jean shorts raw denim Vice normcore, art party High Life PBR skateboard stumptown vinyl kitsch. Four loko meh 8-bit, tousled banh mi tilde forage Schlitz dreamcatcher twee 3 wolf moon. Chambray asymmetrical paleo salvia, sartorial umami four loko master cleanse drinking vinegar brunch. <a href="https://www.pinterest.com" target="blank">Pinterest</a> DIY authentic Schlitz, hoodie Intelligentsia butcher trust fund brunch shabby chic Kickstarter forage flexitarian. Direct trade <a href="https://en.wikipedia.org/wiki/Cold-pressed_juice" target="blank">cold-pressed</a> meggings stumptown plaid, pop-up taxidermy. Hoodie XOXO fingerstache scenester Echo Park. Plaid ugh Wes Anderson, freegan pug selvage fanny pack leggings pickled food truck DIY irony Banksy.

<blockquote>
    We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
    —Anais Nin
</blockquote>
