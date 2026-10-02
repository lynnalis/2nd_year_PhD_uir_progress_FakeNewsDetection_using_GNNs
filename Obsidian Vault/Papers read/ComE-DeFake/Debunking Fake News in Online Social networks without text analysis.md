#### ~={green}**Notes:**=~
* Exploring the feasibility of differentiating between real and fake news by capturing its relationships with other news and users in the network without going through text.
* Generally, in social networks, ~={blue}users are active in community form=~ and ~={blue}propagate news in groups=~ + There's positive correlation between news authenticity and user (user community) credibility,  **but** existing pairwise-oriented methods fail to capture these community-level patterns for FND ==> <mark style="background: #ADCCFFA6;">Employment of Hypergraphs to capture community-level relations between news and users. </mark>
* The users are nodes and news are the hyperedges.
* The framework **ComE-DeFake** (**Com**munity driven hyp**E**rgraph learning to learn high-order relations for **De**bunking **Fake** news in online social networks) works as follows:
    1. Use of hyperedge convolution to capture patterns of news dissemination.
    2. Each user representation embeds credibility with credible or suspicious flag in order to depict (represent) complicated interaction relations among users. 
    3. Aggregation of the credibility of involved users for each piece of news.

#### ~={green}**Framework, promptly:**=~
Down below is the proposed framework:
1. **Global perception of news relations**, which learns high-order and global representations of news using hyperedge learning.
2. **Community driven encoding of user credibility**, capturing community level credibility representations of users through a hypergraph autoencoder with the community layer.
3. **Users-to-News Relation Augmentation**, which integrates user credibility into their involved news to augment news representations.
4. **Detector**, a classifier for identifying real and fake news.

![[Screenshot 2026-03-04 120134.png]]

#### ~={green}**Framework, in details:**=~
First, we construct the attributed hypergraph $\mathcal{G}$ like in the previous paper HyDeFake (nodes set $\mathcal{U}$ users and hyperedge set $\mathcal{N}$ is the news), $\mathbf{X}_{e}$ is the **textual attributes** of the news (**to be replaced with Identity matrix $\mathbf{I}$ since we're not doing any text analysis.)** and $\mathbf{X}_{u}$ is the user attributes.

##### ~={blue}1. Global perception of news relations=~
In order to perceive high-order relations in news, we use a hypergraph convolution network for hyperedge learning of the news. The convolution of the hyperedges goes as follows:
![[Pasted image 20260304121504.png|646]]
For each hyperedge (news), we minimize the cross-entropy loss:
![[Pasted image 20260305133836.png]]
where $y_{i}\in\mathcal{Y}_{tr}$ means that the ground truth labels of the news are in the training data label set with $y_{i}=1$ indicating *fake*, otherwise, it's *real*. $\hat{y}$ denotes the predicted value.
##### ~={blue}2. Community-driven encoding of User Credibility=~
This module aims at leveraging user interactions and community behaviours to evaluate the credibility of users spreading news without news text, given *<mark style="background: #ABF7F7A6;">the group engagement of the users when creating & disseminating news</mark>* and <mark style="background: #ABF7F7A6;">*the positive correlation between user credibility and news authenticity.*</mark>  It goes down like follows:
* ~={cyan}**High-order credibility Encoder:**=~
    Similar to HyDeFake with its User Credibility Channel, we encode high-order interactions between users (node-edge-node transform) using hypergraph convolution, refining the user credibility attributes $\mathbf{X}_{u}$ with user-news-user relations. 
    ![[Pasted image 20260304122801.png]]
* **~={cyan}Collaborative decoder:=~**
    - Here where it differs from HyDeFake at the decoder part, with <mark style="background: #ABF7F7A6;">the absence of news textual embeddings (hyperedge attributes</mark>), a **decoder shouldn't reconstruct neither the original node nor edges attributes**, but instead work on **reconstructing the incidence matrix** representing the structural connections between users and news.
    - The decoder make use of **both** learned representations of nodes $\mathbf{V}$ (users) and hyperedges $\mathbf{E}$ (news) to recover those connections, as follows:
    ![[Pasted image 20260304125027.png]]
    where $p(\hat{\mathbf{H}}(v,e)|\mathbf{V}(v),\mathbf{E}(e))=sigmoid(\mathbf{V}(v),\mathbf{E}(e)^{\top})$ .
    The hypergraph reconstruction loss is minimized by: 
    ![[Pasted image 20260304125651.png|347]]
* ~={cyan} **User-community Self-optimization:**=~
    * To better distinguish between representations of credible and suspicious users in a lower-dim space, we introduce a **community layer** during user high-order relations encoding. 
    * This layer groups the users into clusters representing "<mark style="background: #ABF7F7A6;">credible" or "suspicious" communities</mark> to be used as **soft labels** to supervise the learning process. 
    * The nodes (users) representations are <mark style="background: #ABF7F7A6;">self-optimized by minimizing a clustering objective</mark> as follows:
    ![[Pasted image 20260304152309.png]]
    where $Q$ is the distribution of soft community labels, and $P$ is the target distribution of $Q$. $q_{ic}$ measure **similarity between node embeddings $v_{i}$ and centroid embedding $\mu_{c}$ , like so:** ![[Pasted image 20260304152631.png]]
    with $k$ the number of communities, $p_{ic}$ is the target distribution defined as $\frac{q_{ic}^2 /\Sigma_{i}q_{ic}}{\Sigma_{k} \; q_{ik}^2 / \Sigma_{i}q_{ik}}$. 


##### ~={blue}3. Users-to-News Relation Augmentation:=~
Similar to the concept of Feature fusion in HyDeFake, we amplify the news representations with nodes-to-hyperedge aggregation. 
==> For each piece of news $n_{i}$, we aggregate the credibility embeddings of the subset of users $\mathcal{U}^s_{i}\subseteq\mathcal{U}$ involved in it by mean pooling, to get the **representative user embedding**.
==> we concatenate the news representation $e_{i}$ and the normalized representative user embedding $u'_{i}$.
![[Pasted image 20260305123941.png]]
where $v_{j}$ is the credibility embedding of $j$-th user if $\mathcal{U^s_{i}}$. 
Following the adding operation, we get the **augmented news representation** $z_{i}$.

##### ~={blue}4. Detector :=~
Following the obtention of the final news representations, the task of debunking fake news becomes a <mark style="background: #ABF7F7A6;">hyperedge embeddings</mark>.
==> We pass those embeddings through a softmax layer for news prediction.
$$\hat{\mathcal{Y}}=SOFTMAX(\mathbf{Z})$$ where $\mathbf{Z}$ is the output representations of news and $\hat{\mathcal{Y}}$ is the predicted labels.
==> We optimize the *hyperedge learning*, the *node learning with a community layer*, and define the total loss function:
$$\mathcal{L}=\mathcal{L_{e}}+\alpha\mathcal{L}_{r}+\beta\mathcal{L}_{c}$$ where $\alpha$ and $\beta$ are balance coefficients. 

Down below is the training algorithm of ComE-DEFake:
![[Pasted image 20260305134158.png]]

06/03 ~={red}**Observations:**=~
* No mentioning of what the social context looks like in the paper; what are the user credibility features crawled? Maybe, the same as HyDeFake (user information, which includes ‘‘followers_count’’, ‘‘friends_count’’, ‘‘listed_count’’, ‘‘verified’’, ‘‘statuses_count’’, and ‘‘fav- ourites_count’’).
* Same Hypergraphs used in both papers.