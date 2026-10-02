
# ~={purple}**Fake News features:**=~

![[1-s2.0-S1568494623002533-gr2_lrg.jpg]]

*According to perplexity:*

==> The survey categorizes fake news features into **seven main categories** based on news attributes and discriminative characteristics. These features are extracted from both news content and social context to build effective detection models.​

**1. Linguistic-based Features:**
Capture writing style attributes (words, phrases, sentences, paragraphs) typical of fake news:
- **Lexical**: n-grams, negation/doubt/abbreviation/vulgar words, word novelty
- **Syntactic**: Punctuation count, function words (nouns/verbs/adjectives), POS tags frequency, sentence complexity​
- **Semantic**: Latent topics, contextual clues (via embeddings, LDA)​
- **Domain-specific**: Quoted words, graph frequency, external links​
- **Informality**: Typos, swear words, netspeak, assent words​


**2. Sentiment-based Features:**
Capture emotions/feelings in news:
- **Text polarity**: Positive/negative sentiment​
- **Visual polarity**: Positive/negative images/videos, anxious/angry/sad visuals, exclamation marks​

 **3. User-based Features:**
Properties of accounts creating/spreading fake news (malicious accounts like bots/cyborgs):
- **Individual level**: Registration age, followers count, posts count​
- **Group level**: User ratio, followers/followees ratio​

**4. Post-based Features:**
User responses/opinions to shared news:
- **Post level**: Support/deny opinions, main topic, reliability degree​
- **Group level**: Supporting/contradicting opinion ratios, reliability degree
- **Temporal level**: Changing post/follower counts over time, sensory ratio​

**5. Network-based Features:**
Media/propagation attributes (echo chamber, diffusion):
- Propagation constructions, diffusion methods, density, clustering coefficient​
- Network patterns: Stance, co-occurrence, friendship, diffusion networks​:
    * The stance network is a graph with nodes, edges, nodes showing all the text related to the news, and edges between nodes show similar weights of stances in texts. 
    * The co-occurrence network is a graph with nodes showing users and edges indicating user engagement, such as the number of user opinions on the same news.
    * The friendship network is a graph with nodes showing users who have opinions related to the same news and edges showing the followers/followees constructions of these users. 
    * The diffusion network is an extended version of the friendship network with nodes that indicate users who have opinions on the same news; the edges show the [information diffusion](https://www.sciencedirect.com/topics/computer-science/information-diffusion "Learn more about information diffusion from ScienceDirect's AI-generated Topic Pages") pathways among these users.

 **6. Data-driven Features**
Based on data characteristics:
- **Domain**: Domain-specific/cross-domain knowledge​
- **Concept**: Concept drift detection​
- **Content**: Latent topics, contextual clues (NLP techniques)​

 **7. Visual-based Features:**
Properties of images/videos/links in news:
- **Visual level**: Clarity, coherence, similarity distribution, diversity, clustering score​
- **Statistical level**: Image/video ratios​

**8. Latent Features (Cross-cutting):**
Not directly observable; enhance other categories:
- **Latent textual**: From BERT/ELMo (contextualized), Word2Vec/FastText/GloVe (non-contextualized), knowledge graph embeddings​
- **Latent visual**: Image/video pixel tensors via neural networks​.

These categories enable comprehensive modeling of fake news characteristics like deception intent, malicious accounts, and echo chambers


## **~={purple}Fake News detection techniques:=~**

![[1-s2.0-S1568494623002533-gr3_lrg.jpg]]

*According to perplexity*

The survey classifies fake news detection techniques into **six main categories** (plus hybrid approaches), primarily focusing on content, context, propagation patterns, and multi-label learning.​

**1. Content-Based Detection: Knowledge-based Detection:**
Uses external knowledge sources (knowledge graphs/triples) to fact-check news statements against verified facts.
- Compares extracted subject-predicate-object triples from news with true knowledge graphs.
- Automated (computational) or manual (expert/crowd-sourced) fact-checking.
- Suitable for authenticity verification but struggles with new/unverified information.
​
 **2. Content-base detection: Style-based Detection:**
Analyzes writing style/content patterns distinctive to fake news (e.g., sensationalism).
- **Style representation**: Lexical/syntactic features, embeddings.
- **Style classification**: ML/DL classifiers on style features.
- Effective for immediate detection post-publication but vulnerable to style evolution.
​
**3. Context-based Detection:**
Leverages surrounding social context (sources, publishers, interactions).
- Assesses source credibility (authors/publishers or user networks).
- Uses user profiles, interactions, publisher reliability.
- Best for early detection via spam detection or user behavior analysis.​

**4. Propagation-based Detection:**
Models news diffusion patterns on networks (echo chambers, spread velocity).
- **News cascades**: Temporal sharing trees/sequences.
- **Propagation graphs**: Custom graphs capturing stance/co-occurrence/friendship/diffusion.
- Captures echo chamber effects but requires dissemination data.​

 **5. Multi-label Learning-based Detection:**
Handles multi-class labels (e.g., true/false/unverified) with correlated predictions.
- Style representation/classification, news cascades, propagation graphs.
- Learns joint label distributions across instances.
- Addresses complex labeling but computationally intensive.​

**5. Hybrid-based Detection:**
Combines ≥2 approaches for richer information fusion.
- Content+context, propagation+content, context+propagation.
- State-of-the-art performance by capturing complementary signals
- Examples: GCAN (context+propagation), Bi-GCN (multi-view).​

**Early Detection Notes**: Style/context work best initially; knowledge/propagation unsuitable due to data limitations. GNNs excel in propagation/context hybrids via graph modeling.


## **~={purple}Taxonomies of GNNs:=~**

1. ==Conventional GNNs:==
*They came in as an extension of RNNs to graph structured data.
Their mechanism is based on an information diffusion process where node states (embeddings) are iteratively updated by exchanging information with neighbours .*

**a) How GNNs Process Graph Data?**
* GNNs process graph data by ****passing information between nodes**** through their connections (edges). Each node gathers information from its neighbours in a process called ****message passing****, allowing it to learn representations based on its local structure and features. This iterative information-sharing allows nodes to understand the context provided by surrounding nodes in the graph.*
**b) A typical GNN operates in 3 steps:**
* **Initialization**: Each node is initialized with its feature vector, which could represent properties like age, gender, or molecular weight, depending on the application.
* ***Message Passing:** Over several ****iterations or "layers,"**** nodes exchange information with their neighbours, aggregating data from connected nodes.
* ***Update:** Each node updates its feature vector using aggregated information, often by applying a neural network layer (e.g., a linear transformation followed by a non-linear activation function).

2. ==Graph Convolutional Networks (GCNs):==
* They apply convolutional operations on graph data just like how it's done with images using traditional CNNs. 
* The main idea is to generate a node's representation by aggregating its own features and neighbours' features.

3. ==Graph autoencoders (GAEs):==
* They are unsupervised learning frameworks that encode nodes/graphs into a latent vector space and reconstruct graph data from the encoded information, as follows:
    - The encoder (often a GCN) produces latent node embeddings capturing both structural and content-related information.
    - The decoder reconstructs adjacency or attribute matrices to ensure meaningful embedding learning.

4. ==Spatial-temporal graph neural networks (STGNNs):==
* They are designed for dynamic graphs where node features and graph structures vary over time. they consider spatial dependency and temporal dependency at the same time by combining spatial graph convolutions with temporal models such as RNNs or CNNs to capture temporal dependencies along with spatial graph relations.
* There are two main approaches:
     * ***RNN-based:** the hidden states of STGNNs are passed to a recurrent unit based on graph convolutions. 
     * ***CNN-based:** spatial-temporal graphs are handled recursively by RNN-based approaches, meaning they iterate the propagation process which suggests limitations regarding the propagation time and gradient explosion or vanishing problems. **CNN** approaches solve these problems with *parallel computing* for stable gradients and low memory. 

5. ==Attention-based graph neural networks (AGNNs):==
* AGNNs enhance standard GNNs by integrating attention mechanisms, which dynamically weigh the importance of neighbouring nodes during feature aggregation.
- This adaptive weighting allows AGNNs to focus more on influential nodes or edges that contribute significantly to the detection task.
- AGNNs naturally handle the heterogeneity and complexity inherent in social networks and fake news propagation.
* They are GNNs enhanced with *attention mechanisms* to weigh neighbour contributions adaptively instead of uniform aggregation.
* They get rid of all intermediate FFN layers and all the propagation layers are replaced with an *attention mechanism*, all maintaining the graph structure. 
* The attention mechanism allows learning a dynamic and adaptive local summary of the neighbourhoods to obtain more accurate predictions. 


## **~={purple}GNNs applied for FND:=~**

1. ==Detection approaches based on GNNs:== 
These methods apply a similar set of recurrent parameters to all nodes in a graph to create node representations with better and higher levels.



**a) #[ Continual Learning for Fake News Detection from Social Media ](https://dl.acm.org/doi/10.1007/978-3-030-86340-1_30)

- They exploit GNNs*' capability with non-Euclidean data (graph data in this case) to distinguish propagation patterns of real versus fake news on social media.
    
- Two GNN instances are trained: one on complete data and another on partial data using **continual learning** techniques **(Gradient Episodic Memory (GEM) and Elastic Weight Consolidation)**.
     - *Gradient Episodic Memory (GEM)* is a continual learning algorithm employed to prevent the problem of **catastrophic forgetting** that takes place when training on new tasks degrading performance on previously learned tasks, by maintaining a fixed memory buffer containing representative samples (episodic memory) from previous tasks.
    
     - - When learning a new task tt, GEM:
         - Computes gradients on new task data.
        
         - Also computes gradients on the stored memory samples from previous tasks.
        
         - Modifies the new task gradients such that they do not increase the loss on previous tasks, by projecting gradients to a feasible region that avoids interference.
        
- This preserves old knowledge while allowing learning of new information.
     - *Elastic Weight Consolidation (EWC)* tackles **catastrophic forgetting** in continual learning by protecting important model parameters learned from previous tasks with a regularization term added to the loss function that penalizes large changes in those weights.
- Those continual learning techniques enables **early fake news detection** by adapting to new data without the need to retrain on the whole dataset.
- This method showed outstanding performance without reliance on textual information and efficient training on growing data, **However**, it doesn't fully address *strong forgetting* issues where some prior knowledge can be lost.

****
**b) # [Fake news detection in social media using graph neural networks and nlp techniques: A COVID-19 use-case](https://ceur-ws.org/Vol-2882/paper54.pdf)**

- They focus on detecting malicious users spreading misinformation about COVID-19 and 5G networks conspiracy theories (tweets).
- They employ two strategies (*Content-based fake news detection* and *Context-based fake news* detection to learn representations and classify comments into categories (non-conspiracy, conspiracy, other conspiracies) )
- They manage to achieve a high ROC-AUC score (0.95) mainly due to textual and structural information.
     -**ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)** is a widely used performance metric for evaluating binary classification models. It measures the *model's ability* to distinguish between *positive and negative classes* across *all possible thresholds*, with a score of 1.0 indicating a perfect model and 0.5 indicating random performance.
- **However**,  Content and structure information were used separately, not simultaneously.

**c)[# FANG: leveraging social context for fake news detection using graph representation](https://dl.acm.org/doi/10.1145/3517214)

- FANG (Factual News Graph) is a **context-based** model that focuses on capturing *news context*, including sources, users, interactions and timelines.
- It constructs two **homogeneous subgraphs**: one for **news sources** and one for **users**, both processed by an *unsupervised model* that captures neighbours relations.
- It also uses a **pretrained detection network** for news content extraction as *auxiliary inputs*. 
- It achieves *robustness* even with *limited training data*.
- **However**, features (such as users and their interactions) are extracted **before being fed to FANG**, which may present *errors* regarding text encoding and emotion detection that are passed to FANG. In addition to that, FANG rely on static contextual data that risks to expire (broken hyperlinks, deleted sources...)

**d) [Graph-based modeling of online communities for fake news detection](https://deepai.org/publication/graph-based-modeling-of-online-communities-for-fake-news-detection)

- It constructs a GNN with **heterogeneous input graphs** (multiple types of nodes and edges) . It focuses on analysing online social communities and their influence on fake news spread with no need to rely on user profile data.
- It uses network information in order to capture users' social roles.
- It proposes two GNNs variants: **Hyperbolic** and **Relational** to model user-community hierarchies. 
- It shows improvements compared to conventional GNNs
    - **Hyperbolic GNNs** perform graph learning in hyperbolic space, a non-Euclidean geometry characterized by constant negative curvature, rather than traditional Euclidean space.
     - **Traditional GNNs** embed nodes in **Euclidean space**, which is well-suited for graphs with uniform or grid-like structures but struggles with hierarchical or scale-free graphs, often resulting in high distortion of the graph’s true structure.
     - **Hyperbolic GNNs** embed nodes in **hyperbolic space**, which expands exponentially and naturally fits hierarchical, tree-like, or power-law distributed graphs, enabling more compact and faithful representations of such structures.
     - **Relational GNNs** are designed to handle multi-relational graphs or knowledge graphs where multiple types of edges (relations) exist between nodes.
- **However**, Hyperbolic GNNs perform comparably but not superior to relational GNNs; modeling hierarchical social networks remains challenging.


2.  ==Detection approaches based on CGNs:== 

**a) [[2004.11648] GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media](https://arxiv.org/abs/2004.11648)

* This method works as follows: (content not context (mistake in paper, I presume))
     * (i) The network processes user features, tweet content embeddings, and the propagation structure concurrently:
    {
     * *extract quantified features related to users;
    * *convert words in news tweets into vectors; 
    * *represent aware propagation methods of tweets among users;
    }
    * (ii) capture the correlation between tweet ~~context~~ content and user interactions and between tweet ~~context~~ content and user propagation;
    * (iii) classify tweets as fake or real news by combining all learned representations. 
* This method integrates **dual co-attention mechanisms** with GCNs. 
     * The **first** simultaneously captures **relations between tweet ~~context~~ content and user interactions**. 
     * The **second** simultaneously captures **relations between tweet ~~context~~ content and user propagation.**
*  This multi-level attention captures deep correlations between content and social context, helping to distinguish fake from real news effectively.

**b) [Rumor Detection Based on SAGNN: Simplified Aggregation Graph Neural Networks](https://www.mdpi.com/2504-4990/3/1/5) **(TO READ, MDPI)

* This method aims to calculate the degree of interaction between twitter users for rumour detection.
* It implements a simplified version of GCNs by eliminating the weight matrices.
-  It also adjusts the adjacency matrix to incorporate parent (source tweet) -child tweet (responses or retweets) relationships explicitly.

**c) [Rumor Detection on Social Media via Fused Semantic Information and a Propagation Heterogeneous Graph, KZWANG](https://www.mdpi.com/2073-8994/12/11/1806) **(TO READ, MDPI)

* This method constructed a **heterogeneous graph KZAWNG** for rumour detection by capturing the local and global relationships on Weibo between sources, reposts, and users.
* **KZWANG** is a combination of the news text representation using attention multi-head attention mechanism and propagation representation using CGNs.
* The outputs of the GCN layer and the multi-head attention layer are the inputs of rumour classification.

**d) [Detection of rumor conversations in Twitter using graph convolutional networks | Applied Intelligence](https://link.springer.com/article/10.1007/s10489-020-02036-0) (TO READ, SPRINGER, closed)

* This method constructs two independent GCNs:
     * GCN of tweets (source and reply; tweet i replies to tweet j)
     * GCN of users (interactions among users; user i sent m tweets to user i in conversation )
* These two GCNs are concatenated into one fully connected layer for final classification.

**e) [Rumor Detection by Propagation Embedding Based on Graph Convolutional Network | GraphSAGE | Atlantis Press](https://www.atlantis-press.com/journals/ijcis/125954164) (TO READ, Atlantis Press,open)

* It is a rumour detection method based on propagation detection. 
*  **(NOT VERY WELL EXPLAINED, NEED TO READ PAPER!!!)**


**NOTE!!!**
There are still papers related to CGNs in the the survey, but I prefer to stop in here for the sake of not heavying the presentation as I still have two other detection approaches (AGNNS, GAE). Those papers should be read. 


3. ==Detection approach based on AGNNs:==

**a) [Adversarial Active Learning Based Heterogeneous Graph Neural Network for Fake News Detection |AA-HGNN | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9338358) (TO READ : [2101.11206](https://arxiv.org/pdf/2101.11206) )

* This method constructs a **heterogeneous graph** or a Heterogeneous Information Network (HIN) comprising of different types of nodes (creators, news, subjects) and edges (write, belongs to).
* It employs a **two-level attention mechanism** defined below:
     * 1. **First Level: Attention at Neighbour Node Level**
        - At this level, the model focuses on learning the *importance (weights) of individual neighbours of a given node*.
        - For each neighbour node, the AGNN calculates an attention score that reflects how relevant or influential that neighbour is to the target node.
        - These attention weights are learned dynamically during training, rather than being fixed or uniform, which means the model can prioritize some neighbours over others depending on the task, content, or graph structure.
    * 2. **Second Level: Aggregating Type-Specific Neighbour Weights**
        - The nodes have neighbours of *different types* (creators, news, subjects)
        - The second attention level aggregates the attention scores learned from neighbours of the same type (called type-specific neighbours).
        - AGNNs learn a "schema" or higher-order pattern that combines these type-specific aggregated representations effectively.
        - This schema helps the model optimize the *overall representation of a node by weighing the contributions from each neighbour type appropriately*.
* **Rich semantic an structural information relevant for FND are captured** when considering the **heterogeneous aspect of the graph**, leading to improved performance* compared to homogeneous graph-based methods.

* In addition to the attention mechanism implemented, **AA-HGNN combines adversarial learning and active learning**. 
     * ***Adversarial machine learning*** (AML) is the study of how to attack and defend machine learning (ML) models by exploiting their vulnerabilities. It involves creating "adversarial examples," which are manipulated inputs designed to trick a model into making mistakes, and developing defences to make models more robust against these attacks.
     * ***Active learning*** is a special case of Supervised Machine Learning. It is used to construct a high-performance classifier while keeping the size of the training dataset to a minimum by actively selecting the valuable data points form an unlabelled dataset and requests labels for them.
* AA-HGNN incorporates **active learning** where a **selector module** identifies high-value candidate nodes from the unlabelled set to be annotated and included in the training data.
* The selector is trained in an **adversarial way; The selector and the classifier are competing against each other as follows:
     * The selector looks for the challenging/informative nodes that can "fool" the classifier.
     * The classifier tries to accurately classify both labelled nodes and those selected in an adversarial way.
- After active learning optimization, the classifier produces **the final fake news classifications** on the node embeddings enriched through attention mechanisms.


4. ==Detection approach based on GAEs:==

**[A Graph Convolutional Encoder and Decoder Model for Rumor Detection | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9260014)**

* This method proposes a model that captures textual, propagation, and structural information from news for rumour detection. 
* It includes 3 components:
     * The encoder uses a GCN to represent news text to learn information, such as text content and propagation.
     * The decoder uses the representations of the encoder to learn the overall news structure.
	* Implemented simultaneously as the decoder, The detector also uses the representations of the encoder to predict whether events are rumours.
* The method only focuses on *user-based* (e.g., individual level: registration age, number of followers, and number of opinions posted by users, group level: ratio of users, the ratio of followers, and the ratio of followees ) and *linguistic-based* features (e.g, lexical (wording), syntactic (sentence level), semantic (latent content), domain-specific (domain type)) **ignoring** *network-based* features (e.g., propagation constructions, diffusion methods, and some factors related to the dissemination of news...)
* It also shows high performance in scenarios with limited labelled data.



# ~={purple}**Conclusion and open issues**=~

GNN-based fake news detection is relatively new. Thus, the number of published studies is limited. *(The survey dates of 2023; so of course, there are some new publications in this field in the past two year 2024, 2025)*

1. **Graph-based fake news detection benchmarks** may present an opportunity and direction for future research.
2. **Early detection of fake news** involves detecting fake news at an early stage before it is widely disseminated so that people can intervene early, prevent it early, and limit its harm.
3. The current GNN-based methods have a **static** structure, difficult to adapt in real time, which calls for the construction of **dynamic graphs that are spatiotemporally capable of changing with real-time information.**
4. Majority of existing GNN-based methods work with homogeneous graphs. The use of **heterogeneous graphs** that contain different types of edges and nodes is thus a future research direction.
5.  Absence of a hybrid of propagation, content, and context simultaneous usage in one model, which calls for building GNN models by constructing **multiplex graphs** to represent news propagation, content, and context in the same structure.

# ~={purple}**Challenges and future directions:**=~

1. Deepfake currently poses a significant challenge to tackle in FND
2. Detecting whether the influencers' posts are real or fake is a challenge when they are exposed to possible hacking of their accounts to spread fake news or disinformation.
3. Real-time FND is yet to be addressed in an era where news could be fake at one point and real at another.
4. Constructing benchmark datasets and determining the standard feature sets corresponding to each approach for fake news detection remain challenges.
5. Most GNNs resorts to **undirected graphs** and **edge weights as binary values (0,1)** which is unsuitable in many real-life case. That's why **future studies** can construct graphs with the **weights of edges** as the **actual values representing the relationship among the nodes as much as possible**.
6. For NLP tasks, GNNs have **not represented node features** by capturing the **context of a paragraph or an entire sentence**, and can also overlook semantic relationships among phrases in the sentences. Therefore, for **future directions for improving GNNs** should focus on determining *node features* based on **sentence embeddings** or **significant phrase embeddings**.
7. **NO GNNs** has **simultaneously** considered all the content, context, common sense knowledge and semantic relations which are deemed essential for GNN-based NLP tasks. 
8. So far, GCNs have been limited to a **few layers (2 or 3)** due to **vanishing gradient** occurring.