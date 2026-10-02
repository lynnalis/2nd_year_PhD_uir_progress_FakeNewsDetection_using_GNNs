### ~={purple}**Highlights**=~ https://doi.org/10.1016/j.neunet.2025.107302
* Hy-DeFake models news spreading as a hypergraph, enabling accurate fake news detection.
* It captures user credibility and high-order news-user correlations via hypergraph neural networks.
* Hy-DeFake outperforms nine baselines on four real-world datasets from various domains.
* We verify that news authenticity correlates with user credibility, with fake news forming denser communities.

### ~={green} **What's a Hyper-Graph?**=~
![[Pasted image 20260119132030.png]]
* **Hypergraphs** are generalizations of simple graphs that can connect an arbitrary number of nodes in an edge.
* The **incidence matrix** of a hypergraph demonstrates that multiple nodes are assigned to one edge, enabling the **extraction of high-order information**.
* In this work ~={green}**users**=~ are abstracted as ~={green}**nodes**=~, and a **~={green}hyperedge=~** is formed when multiple users engage with the same piece of news through retweeting.
* Within this hypergraph, ~={green}**node attributes**=~ represent ~={green}user properties related to credibility=~, while **~={green}hyperedge attributes=~** encompass ~={green}news textual content=~.

### ~={green} **Hypergraph construction**=~
![[Pasted image 20260119145121.png]]
We define an attributed hypergraph $\mathcal{G}=(\mathcal{U,E}, X, \mathcal{N})$:
* The **node set** $\mathcal{U}$ is ~={green}the users involved in news spreading=~.
* The **hyperedge set** $\mathcal{E}$ is the ~={green}interactions between users and news=~. **Each hyperedge** $e_{i}$ represents **a piece of news** $i$ and can connect more than two users involved in the spreading of this news.
* $X$ and $\mathcal{N}$ represent **attributes of nodes (users)** and **hyperedges (news)** respectively.
* The **node attributes** of users are ~={green}credibility-related properties=~.
* The **hyperedge attributes** are ~={green}news textual contents=~.

We are dealing with a **Hyperedge classification task**. The hypergraph for FND is:  $\mathcal{G}=(\mathcal{U,E}, X, \mathcal{N,Y})$ where $\mathcal{Y}$ represented the label set $y \in \{0,1\}$ assigned to the news.
### ~={purple}**Proposed method**=~
Hy-DeFake consists of four main parts, utilizing the **~={green}attributed hypergraph as input=~**: 
* **(1) News semantic channel:** it updates hyperedge features by *learning [^1]semantic embeddings of news contents*, as each hyperedge represents a piece of news.
* **(2) User credibility channel:** it learns node features through *embedding both user credibility information and the high-order structural information between news and users*.
* **(3) Mutual information-based [^2]feature fusion:** it incorporates the *semantic embeddings of news* (_i.e._, hyperedges) and the *credibility embeddings of relevant users* (_i.e._, nodes) based on *maximizing their mutual information*.
* **(4) Fake news detection:** it models our task as *hyperedge classification* and classifies the news based on the integrated embeddings.

[^1]: **Semantic Embedding** is a method that utilizes pre-trained word embeddings to replace original words in a sentence with their closest neighbors in embedding space, enhancing the representation of semantic meaning in natural language processing tasks

[^2]: **Feature fusion** refers to the integration of multiple visual cues or features representing different aspects of visual characteristics to create a more comprehensive feature representation for tasks like object detection.


![[Pasted image 20260119153453.png]]

#### ~={blue}**1. News semantic channel:** *Updating hyperedge features*=~
We update textual features of news to contain semantic information using a **pre-trained language model (PLM) $\mathcal{M}$ RoBERTa** like so: $$z^{e}_{i}=\mathcal{M}(n_{i}, \forall n_{i}\in \mathcal{N}) \; \color{blue}(1)$$
where $z^{e}_{i}$ is the updated hyperedge feature of a piece of news $n_{i}$. 
The textual contents of news can be processed as a vector embeddings containing semantic information of the news, all through the LM's fine-tuning. ==> ~={blue}These textual embeddings are taken as the hyperedge features=~ $Z^{e}$. 

![[Pasted image 20260119172352.png]]
#### ~={blue}**2. User credibility channel:** *Learning node features*=~
* Users, being indispensable creators and disseminators of news, exhibit differing behaviors towards real VS fake news. Credible ones propagate the reliable news and refrain from spreading misleading ones.
* There is no explicit user classification as benign or malicious.
* Hy-fake captures **patterns related to user credibility attributes** in an **unsupervised** manner.
* ==> The user credibility channel helps **explore the correlation between users and news they disseminate**, learning not only *user credibility features* but also *high-order complex news-users relations.*
* We adopt a ~={blue}**hypergraph neural network (HGNN) in Autoencoder architecture**=~ to **learn the node features 𝒁,** where 'HConv' is a hypergraph convolution
![[Pasted image 20260120093652.png]]

* The encoder (Hypergraph convolution network) designs a hyperedge convolution operation for learning high-order latent correlation between nodes and hyperedges, by performing a **node-edge-node transform** to ~={blue}capture user-news-user relations=~ and~={blue} refine node features of user credibility=~ with the hypergraph structure.
* The hypergraph convolution is defined by: ![[Pasted image 20260119173547.png]] $\color{blue}(2)$
where:
$\rightarrow W$ learnable parameter, $\sigma$ is the non-linear activation function.
$\rightarrow\Theta$ is the filter, $Z^{(0)}=X$
$\rightarrow D_{e}$ and $D_{v}$ are the diagonal matrices of edges degrees and node degrees.
$\rightarrow H=|\mathcal{U}|\times |\mathcal{E}|$ is the incidence matrix, where $h(u,e)=1$ if the node $u$ is on hyperedge $e$ $(u\in e)$, else $h(u,e)=0$.

%%~={purple} According to the equation above, **the encoder maps the input data to the embeddings of users** which contain **user credibility information** and **high-order correlation between users and news.** =~%%

* The training of this channel being **unsupervised**, the **Decoder maps the embeddings back to reconstruct the input user attributes**, like so:  
       $$\hat{x}^{(1)}_{i}= \sigma(\tilde{W}^{(1)}z_{i}+b^{(1)}),$$$$ \dots,  \; \; \color{blue}(3)$$ \; $$\hat{x}^{(k)}_{i}= \sigma(\tilde{W}^{(k)}\hat{x}^{(k-1)}_{i}+b^{(k)})$$ 
       
     where $z_{i}$ is the $i$th node's latent representation learned by the encoding process equation.
       $\hat{x}^{(k)}_{i}$ is the desired reconstructed attribute of node $i$
       $\{\tilde{W}^{(1)},\dots,\tilde{W}^{(k)},b^{1},\dots, b^{(k)}\}$ are the parameters of the decoder with $k$ layers. 

* The training objective of this hypergraph autoencoder is to **minimize the reconstruction error between the input user attributes and the reconstructed user features.**
* The loss function is defined as: ![[Pasted image 20260120093414.png]] $\color{blue}(4)$
 
 ***==Through the learning process in the two channels outlined above,~={red} Hy-DeFake extracts semantic features of news embedded in hyperedges, along with user credibility features and higher-order relations represented by nodes in the hypergraph.
 =~==***
#### ~={blue}**3. Mutual information-based feature fusion**=~
* Under the assumption that there is **positive correlation between news authenticity and user credibility**, Hy-DeFake learns distinctive representations for the final task by **aligning the associated news and users** in the training process and **separating the non-associated ones**, using a **Mutual information (MI)-based feature fusion module**.
![[Pasted image 20260120125108.png]]
*  We first ~={blue}*fuse*=~ the users features associated with each piece of news to have a ***representative user credibility embedding for that news.***
    The news is propagated by multiple users *(each hyperedge has multiple nodes)*, so we define $\mathcal{U}^{s}_{i}\subseteq\mathcal{U}$ a the ~={blue}subset of users involved in news=~ $i$ . We get those representative embeddings of users $\mathcal{U}^s_{i}$ by ~={blue}aggregating the credibility embeddings of those users by ***element-wise [^3]mean pooling***=~, like so: $$\mathbf{u_{i}}=MEAN\{z_{j},\forall j \in \mathcal{U^s_{i}}\subseteq\mathcal{U} \}  \; \color{blue}(5)$$
    where $u_{i}$ is the representative credibility embeddings of users involved with news $i$. 
    $z_{j}$ is the credibility embedding of $j$th user in $\mathcal{U}^s_{i}$.
    We end up with a **representative user credibility matrix $\mathbf{U}$ per news article.


* Next, we~={blue} *combine* =~the *news semantic embeddings* with *these representative user credibility embeddings* using an **MI loss**.~={purple} ==>*Successful integration of the semantic features, credibility features and high-order relationships.*=~
The loss function is **maximized** like so: 
![[Pasted image 20260120113220.png]] $\color{blue}(6)$
where:
$t$ is the number of news.
$z^e_{i}$ and $u_{i}$ are the embedding of the news $i$ and the representative embedding of users spreading the news $i$, respectively. 

==>*~={purple}The term $\mathcal{I}(z^e_{i};\mathbf{u_{i}})+\mathcal{I}(\mathbf{u_{i}};z^e_{i})$ calculates the **symmetric mutual information** between he news embedding and the representative user embedding. Including **both directions** ensures that the **model captures the relationship well from both user and news perspectives.**=~*

The **mutual information** is computed like follows:
![[Pasted image 20260120101945.png]]
where:
$\rightarrow p(z^e_{i},\mathbf{u_{i}})$ is The joint probability distribution of the news features and corresponding representative user features.
$\rightarrow p(z^e_{i})$ and $p(\mathbf{u_{i}})$ are The marginal probability distributions of the news and user features, respectively.
$\rightarrow$ The symbol $\mathbb{E}$ represents the **expectation operator**, which calculates the average value of the term inside the brackets over a probability distribution

==>*The mutual information can be read as: $\mathcal{I}(z^e_{i};\mathbf{u_{i}})$ is equal to the Expected value of $\log \frac{p(z^e_{i},\mathbf{u_{i}})}{p(z^e_{i})p(\mathbf{u_{i}})}$ under the assumption that $(z^e_{i},\mathbf{u_{i}})$ is sampled according to $(p(z^e_{i},\mathbf{u_{i}}))$.*

 ***==The ~={red}maximization of the loss function=~ allows Hy-DeFake to enhance mutual information in two channels, facilitating the ~={red}sharing of positive relations between credible users and real news=~, as well as~={red} uncredible users and fake news=~, respectively.==***

#### ~={blue}**4. Fake news detection:**=~
* Following the ~={blue}fusion=~ of news semantic information and user credibility information, we get ~={blue}integrated embeddings=~ that can be regarded as **hyperedge embeddings**.
* Each hyperedge represents a piece of news, hence, detecting fake news becomes a~={blue} **hyperedge classification**=~.
![[Pasted image 20260120133953.png]]
* These hyperedges embeddings serve as *input* to the Hyperedge classifier; They go through an MLP followed by a SOFTMAX layer for final news prediction, formulated like so: $$z^{0}_{i}=f(W'^{(k)}(z^e_{i}\oplus\mathbf{u}_{i})+b'^{(k)}), k=1,\dots, K$$
  $$\hat{y}=SOFTMAX(z^0_{i})$$
where:
$\rightarrow z^{(0)}_{i}$ is the output embedding of news $i$ after MLP and $\oplus$ us the concatenation operator.
$\rightarrow W'^{(k)}$and $b'^{(k)}$ are the parameters on layer $k$ and $\hat{y}$ is the predicted label.

* For each piece of news (hyperedge), the objective is to **minimize the cross-entropy loss**:
$$ \mathcal{L}_{d} = -y\log\hat{y}-(1-y)\log(1-\hat{y}),  \; \color{blue}(10)$$ where $y\in\{0,1\}$ is the ground truth of label of the news.

==> In Hy-DeFake, we~={blue} combine the three losses=~ (user credibility encoding, MI based fusion, hyperedge classification)~={blue} into one final loss=~ to be optimized: 
$$ \mathcal{L}=\mathcal{L_{rec}} + \mathcal{L_{d}}+\alpha\mathcal{L_{mi}}  \; \color{blue}(11)$$
where $\alpha$ is used to control the balance of objectives.


 ***==Down below is the algorithm behind Hy-DeFake training:==
 ![[Pasted image 20260120132652.png]]
 

[^3]: **_Mean pooling** is a method to get an average of the vector embeddings generated after you encode chunks.




 