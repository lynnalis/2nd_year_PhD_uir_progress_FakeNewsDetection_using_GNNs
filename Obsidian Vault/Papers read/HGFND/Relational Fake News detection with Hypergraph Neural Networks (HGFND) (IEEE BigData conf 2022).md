
### **~={yellow}Highlights:  =~**[Nothing Stands Alone: Relational Fake News Detection with Hypergraph Neural Networks | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/10020234)
* Shifting from pairwise to **high-order group-wise** interactions among news pieces using a Hypergraph structure.
* Constructing a Hypergraph with **three distinct types of hyperedges**; **User** for connecting news shared by the same user, **Time** for connecting news shared within a proximal time range (same day/hour), and **Entity** for connecting news sharing specific entities (organization, people, event,...)
* Employing **dual-level attention mechanism**; **node-level** to identify important news nodes for forming hyperedge representations, and **hyperedge-level** to select most informative hyperedges for learning specific news representations.

### **~={yellow}Preliminaries: =~**

![[Pasted image 20260203140832.png]]
##### **~={purple}a) Propagation tree:=~**
* **A propagation tree is used to show the *cascade of news spreading on a social network through user engagement, such tweets and retweets*.** 
* For each single news piece, we have a propagation tree.
* Given $N$ number of news, the set of features for news contents $\mathcal{X}=\{x_{i}\}^N_{i=1}$ and user engagement associated to news piece $x_{i}$, $\mathcal{U}_{i}=\{u_{t_{j}}\}^{|\mathcal{U}|}_{j=1}$, where $t_{j}$ is the sequence of timestamp of a user's engagement such as tweet and retweet, A **propagation tree** is defined as $P_{i}=(\{x_{i}\}\bigcup \mathcal{U}_{i},\mathbf{A})$ where $\mathbf{A}$ is an adjacency matrix estimated by time-inferred diffusion.

##### **~={purple}b) Hypergraph:=~** (already explained in Hy-DeFake notes.)

![[Pasted image 20260203151305.png]] 
##### **~={purple}c) Problem statement:=~**
* For $N$ number of news, the goal is to predict the label $\mathcal{Y}=\{y\}^N_{i=1}\subset \{O,1\}^N$. 
* The news corresponds to a node in the hypergraph $\mathcal{G}=\{\mathcal{V,E}\}$.
* The hypergraph connects nodes (news) through hyperedge based on the news content $\mathcal{X}$ and their social context obtained from the propagation trees $\mathcal{P}=\{P_{i}\}^N_{i=1}$.


### **~={yellow}Methodology: =~**
##### **A. Hypergraph construction for News relations:**
When constructing the graph, the authors relied on the [FakeNewsNet dataset]([1809.01286](https://arxiv.org/pdf/1809.01286)) [(this is the github repo]([KaiDMML/FakeNewsNet: This is a dataset for fake news detection research](https://github.com/KaiDMML/FakeNewsNet?tab=readme-ov-file))  which contain **both news content** and **user engagements**. The built structure contains news pieces via **3** types of hyperedges:
* ~={green}*User hyperedge* (social context, Shared User ID between news pieces): =~ we retrieve unique User IDs from tweets and retweets within the propagation trees. If different news pieces share the same user ID (same person shared them), they are connected to a single User hyperedge.
* ~={green}*Time hyperedge (temporal context, News shared in proximal time range)*: =~ We assume that **similar news emerge and disseminate within proximal time range.** We rely on the creation of timestamps of tweets and retweets associated with the news, rounded to the nearest day or hour, The news pieces shared within a specific time windows are connected.
* ~={green}*Entity Hyperedge (content context, Shared entity between news contents)* :=~ This hyperedge links news pieces with similar topics by identifying shared entities in the text. The entity recognition tool (*spacy*) is used to extract those entities (organizations (ORG), people (PER), products (PROC), creative works (WORK OF ART), nationalities or political groups (NORP), locations (LOC), and named events (EVENT))
==>At last, the **incidence matrices of the three resulted hypergraphs** $H_{user}, H_{time}, H_{entity}$ are **concatenated** to have a unified structure $H$ ~={red}before the model training begins.=~

##### **B. Leveraging news relations for FND:** *The framework*
![[Pasted image 20260203152751.png]]

~={green}**1. Encoding propagation tree:**=~
This module serves as an **initialisation mechanism for the HGFND model** and generate comprehensive **initial node representations** $v^{0}_{i}$ for the hypergraph by *combining news content with social context derived from how the news spread*.
The process goes as follows:
* Each propagation tree is taken in as an input and encoded using a GNN, particularly, **GraphSAGE**. 
* From the GNN output, we **extract the root representation** $\bar{x}_{i} \in \mathbb{R}^F$, $F$ is the size of the input feature matrix, like so: $$\bar{x_{i}}=ROOT(GNN(P_{i}))$$
==> This vector captures the propagation patterns associated with the specific news piece.
* For the model to consider both the propagation pattern and the news text itself, we **concatenate** the extracted root representation $\bar{x}_{i}$ and the original news content features $x_{i}\in\mathbb{R}^F$ via **skip-connection.
* This combined feature vector is passed through a non-linear activation function $\mathrm{Re}LU$ and fully connected layer. This projects the data into the hidden dimension size $d$, resulting in the initial node representation $v_{i}^0\in\mathbb{R}^d$, like so: $$v_{i}^0=f(\sigma(x_{i}\oplus\bar{x}_{i}))$$
~={green}**2. Hypergraph attention network:**=~
    *==2.1. Node-Level Attention for hyperedge representation:==*
        - The primary goal of node-level attention is to ~={red}aggregate information from individual news nodes to form a unified representation for the hyperedges they belong to.=~ 
        - We look at highlighting the nodes that are most **important** for defining the hyperedge's representation.
        - In order to determine the importance of a specific node $v_{k}$ to a specific hyperedge $e_{j}\in \mathcal{E}$, we calculate an **attention coefficient** $\alpha_{jk}$ like so:
            - We first **transform** the feature vector of the node form the previous layer $v_{k}^{l-1}$ using a trainable weight matrix $W_{1}$. We apply $LeakyRELU$ activation function to this transformed vector, like so: $h_{k}=LeakyRELU(W_{1}v^{l-1}_{k})$
            - We use a **context vector** $a_{1}$ to measure the similarity/importance of the node's hidden state. The resulting scores are **normalized using Softmax function**, like so: $$\alpha_{jk}=\frac{\exp(a_{1}^Th_{k})}{\Sigma_{v_{p}\in e_{j}}\exp(a_{1}^Th_{p})}$$
        - Finally, once the attention coefficients are computed, we generate the hyperedge representation $e_{j}^l$ of the current layer $l$ by calculating a **weighted-sum** of the features of all nodes connected to the hyperedge, using the attention coefficients $\alpha_{jk}$ as weights, then applying a **non-linear activation function** $\sigma$ to this aggregated sum, like so: $$ e_{j}^l=\sigma(\Sigma_{v_{k}\in e_{j}} \;\alpha_{jk}W_{1}v_{k}^{l-1})$$
    ==2.2. Hyperedge-level Attention for Node representation:==
        - The primary goal is to~={red} update the representation of individual news nodes by aggregating information from the various groups (hyperedges) they belong to.=~ 
        - We calculate an **attention coefficient** $\beta_{ij}$ to measure the importance of a specific hyperedge $e_{j}$ to the node $v_{i}$, like so:
            - We compute a **hidden state** $r_{j}$ by concatenating both the transformed features of the hyperedge $W_{2}e^l_{j}$ and the node $W_{1}v^{l-1}_{i}$ and passing them through **LeakyRELU** activation function: $$r_{j}=LeakyRELU([W_{2}e_{j}^l\oplus W_{1}v_{j}^{l-1}]) $$
            - A **context vector** $a_{2}$ is applied to this state $r_{j}$ and the results are **normalised** using softmax function across all hyperedges $\mathcal{E}_{i}$ connected to that specific node $v_{i}$: $$\beta_{ij}=\frac{\exp(a_{2}^Tr_{j})}{\Sigma_{v_{i}\in e_{j},e_{j}\in \mathcal{E}_{i}}\;\exp(a_{2}^Tr_{i})}$$
            - Once the importance weights are computed, we update the node's representation $v_{i}^l$ for the current layer $l$, by calculating the **weighted-sum** of all hyperedge representations connected to that node, using the attention coefficients $\beta_{ij}$ as weights, then applying a **non-linear activation function** $\sigma$ to this aggregate sum, like: $$v^l_{i}=\sigma(\Sigma_{e_{j}\in\mathcal{E}_{i}}\; \beta_{ij}W_{2}e^l_{j})$$

~={green}**3. News (node) classification:**=~
This last part of the framework maps the high-level representations of news (nodes) learned by the hypergraph attention network into a binary prediction: **determining whether a piece of news is "~={yellow}fake=~" or "~={yellow}real=~"**. 
* The **final node representation** $v^L$ ($L$ is the final layer) is taken as input and **passed through a fully connected layer** with a trainable weight matrix $W_{3}\in\mathbb{R}^{d\times2}$ and a bias $b$, all to project the high-dimensional node features into a 2-dimensional space (2 possible classes). 
* The output is passed through a **softmax function** to convert the raw logits into probabilities $\hat{y}_{i}$, indicating the likelihood of the news node $i$ belong to the fake or real class.$$\hat{y}_{i}=Softmax(f(W_{3}v^L_{i}+b))$$
* During training, the module minimizes the error using **Negative Log Likelihood** loss function.
*$$ \mathcal{L} = \Sigma _{y_{i}\in Y_{train}}(-y_{i}\log\hat{y}_{i}-(1-y_{i})\log(1-\hat{y}_{i}))$$
* It's important to note that the framework works in a **semi-supervised setting**. 
* It refers specifically to a **transductive node classification** approach. This method is designed to maximize the utility of limited data by allowing the model to "see" the entire network structure during training, even though it only knows the "answers" (labels) for a small subset of the news.

Down below is the algorithm behind the HGFND.
![[Pasted image 20260203185549.png]]
