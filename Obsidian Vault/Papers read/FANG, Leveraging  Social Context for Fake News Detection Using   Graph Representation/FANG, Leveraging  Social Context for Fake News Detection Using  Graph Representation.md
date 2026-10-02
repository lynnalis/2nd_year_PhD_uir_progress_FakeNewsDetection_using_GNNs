# **Introduction**
## ~={orange}==Preliminaries:===~
*  **Stance detection is the task of automatically identifying the opinions of an individual or community on a specific topic.**
* The AGGREGATE function is **a primary operation within the message passing mechanism used in Graph Neural Networks**. Its purpose involves taking a set of feature vectors from a node's neighbors, which can vary in size, and condensing them into a single, representative vector.

## ~={red} ==Intro:===~
* Textual and visual features have been widely used to model news article contents, by feature extraction, unsupervised semantics encoding, or learned representation.
* Fact-checking approaches for textual claims based on **textual evidence** (external knowledge databases)  are not **easily applied to claims about images and videos.**

* There have been works that explored **contextual features of the news-dissemination process**. *There were **distinctive engagement patterns** when social users face **fake VS real news***.
    * ~={red}**For the Fake news:**=~ There was **negative sentiment** towards the appalling content of the fake news, then there are **denial posts** that question the credibility of the news, all leading to a stabilization of the distribution afterwards with virtually no support.
    * ~={cyan}**For the real news:**=~ There was **moderate engagement** followed by **supportive posts** with **neutral sentiment** that *stabilize quickly*.
![[Pasted image 20251202130002.png]]

* Other works proposed ***partial representation of social context*** with ~={cyan}(news, sources, users)=~ as major~={cyan} entities=~ and~={cyan} (stances, friendship and publication)=~ as major ~={cyan}interactions=~; however little work's yet to be put in the representation's quality, the entities/interactions modelling.

* The social Context of news dissemination can be represented as a **heterogeneous network** where **nodes are social entities** and **edges are their interactions.** 
* Down below is a *graph representation of social context:*

![[Pasted image 20251202130926.png]]
* Above in the figure, There are the *homogeneous edges (user-user relationship, source-source citation)*, *heterogeneous edges (user-news stance expression, source-news publication)* and *high-order proximity (e.g., users that consistently support or deny certain sources)*.
* This allows the **representation of heterogeneous entities** to be **dependent**, leveraging not only *fake news detection* but also related tasks such as *malicious user detection* and *source factuality prediction.*

**" The PAPER focuses improving contextual fake news detection by enhancing the representations of social entities."**


# **Related works**

* ~={blue}**Contextual FND:**=~
    * *Euclidean approaches:* The social context is represented as a flat vector/ matrix of real numbers. The social entity features are learned via an Euclidean transformation. 
    * **However**, given the **heterogeneous aspect of the social context network, Euclidean representations may not be the best fit**; *Works that used users attributes (demographics, news preference, social features (number of followers/friends)) did not capture the user interaction landscape, that is, what kind of social figures they follow, which news topics they favour or oppose, and so forth.
    * In FANG graphical representation, node variables are no longer constrained by the **independent and identically distributed assumption**. Their each other's representation in reinforced via **edge interactions**.
    * *Non-Euclidean/Geometric approaches:* Given the limitations presented above, researchers explored *geometric approaches* and managed to generalize the idea of using **the social context** by developing **representations that capture structural features about the entity**, when **modelling a target user or the news source network**.
    * Some works on citation source network, propagation network, and rumour detection proposed **models** optimized for **FND only** **without considering representation quality**, exposing their **lack of robustness to limited data** and their **inability to be generalized to other downstream tasks.
    
* ~={blue}**GNNs**:=~
     * CGNs were among the first methods that effectively applied convolutional filters, yet, they impose a **large memory footprint** when storing the adjacency matrix.
     * They **can't be adapted to heterogeneous graph**  (nodes and edges with different labels shows different information propagation)
     * They **don't guarantee generalizable representations for unseen nodes in evolving graphs** and are **transductive; they require inferred nodes to be present at training time**.
     * [Transductive and Inductive learning?](https://perso.isep.fr/pconde/Publi-2022-ASONAM.pdf) These two concepts are related to the choice of neighborhood. 
         * In the transductive setting, the neighbours of the training nodes can belong to the validation and test sets; in this case, only their features are known at training time.
         * In the inductive setting, the neighbours of the training nodes are restricted to other training nodes.

* ~={blue}**Chosen graph in the paper?**=~
     * They chose to work with [GraphSAGE](https://snap.stanford.edu/graphsage/) ([[1706.02216] Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)), an **inductive** framework that leverages node attribute information to efficiently generate representations on previously unseen data.
     * Instead of training individual embeddings for each node, a function is learned to generate embeddings by sampling and aggregating features from a node's local neighborhood.


# **Methodology**

### **a) FND with social context**
Down below is the definition of the social context graph G with entities and interactions:
![[Pasted image 20251203100700.png]]  
The following table summarizes the types on interactions (homogeneous and heterogeneous) within the social context network.

![[Pasted image 20251203150949.png]]
Given a social context graph G=(A,S,U,E), **context-based FND** is defined as the *binary classification* task to predict whether a news article a is fake of real:
![[Pasted image 20251203151546.png]]
### **b) Graph construction from social context**
* <mark style="background: #FFB8EBA6;">News articles: </mark>
     The **news article feature $x_{a}$= concat\[TF.IDF vector** from the text body of the article, **semantic vector** from weighting the GloVe pretraining embeddings of each word by their TF.IDF score]==
* <mark style="background: #FFB8EBA6;">News sources:</mark>
     Similar to the article representations. **The source feature vector ==$x_s$** = **concat\[TF.IDF vector**, **semantic vector** derived from the words in *Homepage* and *About Us*].==
* <mark style="background: #FFB8EBA6;">Social users:</mark>
     **User feature vector ==$x_u$**= **concat\[TF.IDF vector** and a **semantic vector** derived from the textual description in user profile]==
* <mark style="background: #FFB8EBA6;"> Social interactions:</mark> 
       * Each pair of social actors $(v_{i}, v_{j})\in A \cup S \cup U$, we add an edge $e=\{v_{i},v_{j},t,x_{e}\}$ to the list of social interactions $E$ if they are linked via an interaction type $x_{e}$.
        ==* followership* $\rightarrow$ user $u_{i}$ follows user $u_{j}$==
        ==* *publication* $\rightarrow$ news article $a_{i}$ is published by source $s_{j}$==
        ==* *citation* $\rightarrow$ Homepage of source $s_{i}$ contains hyperlink to source $s_{j}$==
       * The $t$ is for the time for time-sensitive interactions (*publication* and *stance*) to be recorded with respect to the article's earliest time of publication. 
* <mark style="background: #FFB8EBA6;"> Stance detection:</mark>
     * In general, ***Stance detection is characterizing the viewpoint of a text with respect to another one***. In FND context, we're interested in the stance of a user reply/social media post with respect to the title of news article. 
     * **Four stances** are considered~={green} **(neutral support, negative support, deny, report)=~. 
     * A **social media post** is classified as a **Verbatim report** of the **news** if it **matches the article title after cleaning** (removal of stop words, urls, emojis, punctuation...)
     * We train a **stance detector** on the remaining posts to classify them as *support* or *deny* using a *constructed annotated dataset for stance detection between social media posts and news articles of 2597 labelled source-target sentences pairs from 31 news events.* 
     * (1st order: **reference headline–related headline or the head line–related post sentence pairs**) $\rightarrow$ For each reference headline, a list of related headlines and posts was given to annotate as either **supporting it or denying it**. 
     * (2nd order: **related headline–related post sentence pairs**) $\rightarrow$ . If such a pair expressed a similar stance with respect to the reference headline, it's a **support** stance, otherwise, it's a **deny**.
    ![[Pasted image 20251203173104.png]]
    * **RoBERTa-large** was finetuned on that data for the stance detection.
    * For the **support subclassification (with neutral sentiment, with negative sentiment), **RoBERTa-large-based sentiment classifier** was fine-tuned of the dataset of [Yelp Review Polarity](https://www.kaggle.com/datasets/irustandi/yelp-review-polarity)


### **c) Factual News Graph (FANG) framework: 

![[Pasted image 20251217150924.png]]

#### *~={blue} **Representation learning**=~
FANG derives the representation of each social entity using *GraphSAGE's node-encoding function.

* **Important:** Down below is the algorithm from the GraphSAGE Paper, behind the node-encoding/node embedding generation.
![[Pasted image 20251204094415.png]]
Let's explain the algorithm above. 
* $k$ denotes the current step/depth of the search in the outer loop, $h_{k}$ is a node's representation at that time step. 
* The representations at $k=0$ are defined as the input node features.
*  Each node $v \in V$  aggregates the representations of the nodes in its immediate neighborhood $\{h_{u}^{k-1}, \forall u \in N(v)  \}$ into a single vector $h_{N(v)}^{k-1}$. 
* After that, GraphSAGE concatenate the node's current representation $h_{v}^{k-1}$ with the aggregated neighborhood vector $h_{N(v)}^{k-1}$. 
* The concatenated vector is fed through a Fully connected layer with non linear activation function $\sigma$ , which transforms the representations to be used in the next time step of the algorithm (i.e., $h_{v}^{k}, \forall v \in V$).
* The final representation output at depth $K$ is denoted as: $z_{v}\equiv h_{v}^{K}, \forall v \in V$.
* The aggregation of the neighbour representations can be done via a variety of aggregator architectures.


~={blue}****Back to FANG:***=~
* The structural representation of any user or source node can be defined as $z_{r}=GraphSAGE(r), z_{r} \in \mathbb{R}^{d}$ with $d$ *the structural embedding dimension.*
* For ~={green}news nodes=~, we enrich their structural representation with ~={green}user engagement temporality=~ by learning an ~={green}**aggregation function** =~$F(a,U)$ that maps news $a$ and its engaged users $U$ to a temporal representation $v_{a}^{temp}$ that captures $a$'s engagement pattern. 
    * The aggregating model (The AGGREGATOR) chosen here is the ~={green}**BI-LSTM with attention**=~ for time-sensitivity.
    * The LSTM input is a user-article engagement sequence $\{e_{1}, e_{2},\dots, e_{|U|}\}$. 
    * Let $meta(e_{i})\in \mathbb{R}^l=(time(e_{i}),stance(e_{i}))$ be the concatenation of $e_{i}$'s elapsed time since the news publication and a one-hot stance vector.  
    * Each engagement $e_{i}$ has the following representation: $x_{e_{i}} = (z_{v_{i}}, meta(e_{i}))$, where $z_{v_{i}}=GrapheSAGE(U_{i})$.
    * The Bi-LSTM encodes the engagement sequence and outputs two sequences of **hidden states**:
        **~={green}*forward one=~** (from beginning of sequence): $H^f=h_{1}^f, h_{2}^f,\dots,h_{n}^f$ 
        **~={green}backward one=~** (from the end of the sequence):  $H^b=h_{1}^b, h_{2}^b,\dots,h_{n}^b$ 
    * Let $w_{i}$ be the **attention weight** paid by the Bi-LSTM encoder to the **forward $h_{i}^f$ and backward $h_{i}^b$ hidden states.** They derive from the **similarity of the hidden state and the news features**. 
        $\rightarrow$ How relevant the engaging users are to the content, the particular time and the stance of the engagement. 
    ![[Pasted image 20251205151249.png]]
    where: 
    $\rightarrow l$ is *the meta dimension*
    $\rightarrow e$ is *the encoder dimension*
    $\rightarrow M_{e} \in \mathbb{R}^{d \times e}$ and $M_{m}\in \mathbb{R}^{l \times 1}$ are *optimisable projection matrices for the engagement and meta features, shared all across engagements.*
    * We use $w_{i}$ to compute the forward and the backward weighted feature vectors as $h^f=\sum ^n_{i}w_{i}h^f_{i}$ and $h^b=\sum ^n_{i}w_{i}h^b_{i}$ respectively.  
    * We concatenate the forward and backward representation vectors to have the ~={green}**temporal representation** =~ $v^{temp}_{a} \in \mathbb{R}^{2e}$ for article $a$, setting $2e=d$ combines the temporal and the structural representations as $$z_{a}=v^{temp}_{a} + GraphSAGE(a)$$

#### ~={blue}**Unsupervised proximity loss**:=~ 
 * It comes from the hypothesis that **closely connected social entities often behave similarly**, all motivated by the *echo chamber*
 $\rightarrow$ Intercited news media sources publish news of similar content/factuality.
 $\rightarrow$ Social friends express similar stance to news article(s) of similar content.
 
 ==*FANG should assign these close entities a set of proximal vectors in the embedding space, whereas the representations of disparate entities should be distinctive.

* The concerned social interactions in the Graph are: **user-user friendship** , **source-source citation** and **news-source publication**. 
* The social context graph is divided into *two subgraphs* : **news-source subgraph** and **user subgraph**. 
* For each subgraph $G'$, we have the following ~={blue}*Proximity loss*=~ : 
![[Pasted image 20251205151229.png]]
where:
$\rightarrow z_{r}\in \mathbb{R}^d$ is *the representation of an entity $r$*
$\rightarrow P_{r}$ is the set of nearby nodes (*positive set*) of $r$ obtained with a **fixed length random walk**
$\rightarrow N_{r}$ is the set of disparate nodes (*negative set) of $r$ derived with **negative sampling**
$\rightarrow q$ is a weighting vector.


#### ~={blue}**Self-supervised stance loss**=~
* It derives from the hypothesis for the **user-news interactions**; **if a user expresses a stance to a news article, their respective representations (user & article) should be close**.
* For each stance $c$, we learn:
    * *a **user projection function**: $\alpha_{c}(u) = A_{c}z_{u}$
    $\rightarrow$ maps user representation $z_{u} \in \mathbb{R}^d$ to a representation in the stance space $c$ of $\mathbb{R}^{d_{c}}$
    * *a **news article projection function**:* $\beta_{c}(a)=B_{c}z_{a}$ 
    $\rightarrow$ maps news article representation $z_{a} \in \mathbb{R}^d$ to a representation in the stance space $c$ of $\mathbb{R}^{d_{c}}$
* We then compute the **similarity score of user $u$ and news article $a$ in the stance space** $c$ of $\mathbb{R}^{d_{c}}$ as $\alpha(u)^\intercal \beta(a)$.
* If $u$ expresses stance $c$ to $a$ , the score is **maximized**, otherwise **minimized**.
* Down below is the *stance classification objective*, optimizing the ~={blue}stance loss=~.
![[Pasted image 20251205154357.png]]

#### **~={blue} Supervised Fake News loss =~**
To predict whether news article $a$ is fake or not: 
   * we get its *contextual representation* by concatenating *its representation* and *the structural representation* of its source which is $v_{a}=(z_{a}, z_{s})$.
   * The contextual representation of news article $a$ is input to a **fully connected layer** which outputs are $o_{a}=Wv_{a}+b$ where $W\in \mathbb{R}^{2d}$ and $b \in \mathbb{R}$ are the weights and biases of the layer. 
   * The output $o_{a}$ is passed through a **sigmoid function** $\sigma(.)$ and trained using **cross-entropy- based** *fake news loss* $\mathcal{L}_{news}$: $$ \mathcal{L}_{news} = \frac{1}{T}\sum_{a}\{y_{a}.\log(\sigma(o_{a}))+(1-y_{a}).\log(1-\sigma(o_{a}))\}$$
   where: $T$ is the batch size, $y_{a}=0$ if $a$ is fake, $1$ otherwise.


**The final total loss is defined by linearly combining these three component losses:** 
$$ \mathcal{L}_{total}= \mathcal{L}_{prox} + \mathcal{L}_{stance} + \mathcal{L}_{news}$$



# **Experiments**

* The experiments were done on a Twitter dataset collected for rumour classification and FND [KaiDMML/FakeNewsNet: This is a dataset for fake news detection research](https://github.com/KaiDMML/FakeNewsNet) .
* For each article, we collect:
    * its source
    * list of engaged users
    * the tweets of those engaged users
* The dataset also includes twitter profile description and list of twitter followees of a given target user.
* Additional data was crawled regarding **media sources**:
    * Content of their *Homepage* and their *About us* page
    * Frequently cited sources on their *Homepage*
* Whether an article is fake or not is based on two fact-checking websites: [Snopes.com | The definitive fact-checking site and reference source for urban legends, folklore, myths, rumors, and misinformation.](https://www.snopes.com/) and [PolitiFact](https://www.politifact.com/)
* The performance of FANG was benchmarked against competitive models:
    * **content only model** (SVM model on TF.IDF feature vectors made of news content)
    * **Euclidean contextual model** (CSI : fundamental yet effective recurrent encoder that aggregates the user features, the news content, and the user–news engagements.)
    * **GCN graph learning framework**
* The importance of modeling temporality was also studied by experimenting on two variants of CSI and FANG: 
    * time-insensitive CSI (-t) and FANG(-t) without $time(e)$ in the engagement $e$'s representation $x_{e}$.
    * time-sensitive CSI and FANG with $time(e)$.
    ![[Pasted image 20251218115526.png]]
* All context-aware models perform better than the non-context one (Feature SVM) ==>Considering the social context is helpful for FND.
* Both time-sensitive CSI and FANG perform better than their time-insensitive versions. ==> The importance of modeling the temporality of news spreading.
* The two graph-based models (FANG and CGN) perform better than the Euclidean CSI (-t). ==> Effectiveness of the social graph representation. 
* **~={green}C/C: FANG model outperforms the other context-aware, temporally-aware and graph-based models.=~**
    