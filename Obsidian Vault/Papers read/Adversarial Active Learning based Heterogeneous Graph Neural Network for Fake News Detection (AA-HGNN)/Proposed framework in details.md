
**A**dversarial **A**ctive Learning based **H**eterogeneous **G**raph **N**eural **N**etwork or **AA-HGNN** consists of two major components: ~={yellow}HGAT-based classifier=~ and ~={yellow}HGAT-based selector=~, with HGAT being Hierarchical Graph Attention Neural Network. 
#### ~={blue} **A) Model Overview**=~

![[Pasted image 20251222162233.png]]
* The news-HIN $\mathcal{G}$ is the input of the classifier. 
* $h^L$ : initial feature of a labelled node and $h^U$ : initial feature of an unlabelled node.
* The~={yellow} HGAT-based classifier=~ is trained with both labelled and unlabelled data to **~={yellow}predict labels $\{\hat{y}\}$ for unlabelled news article nodes=~.** 
* Pairs of **labelled nodes and their ground-truth labels** $\{y\}$ are considered **positive samples** whereas the **unlabelled nodes and their predicted labels** $\{\hat{y}\}$ are seen as **negative samples**.
* A part of these positive and negative samples is used to train the ~={yellow}HGAT-based selector=~.
* After training, he HGAT-based selector outputs the **confidence $\mathcal{P}$ of pairs in test set.**
* Based on that confidence $\mathcal{P}$, the selection strategy takes in a **set of high-value unlabelled nodes** as candidates of size $k$ to be **labelled by experts**.
* These candidates are moved to the training set before next round of optimization. 

**Hierarchical Graph Attention Neural Network (HGAT) is the basis of the classifier and the selector.**
#### ~={blue} **B) HGAT explained**=~
![[Pasted image 20251223123247.png]]

HGAT employs **two-level attention mechanism:**
1. ~={cyan}**Node-level attention**:=~ 
    * It learns the *importance of neighbours of the same type* respectively for each news article node $n_{i} \in \mathcal{N}$ and then *aggregates the representation of same-type neighbours to have a schema node.* 
    * The inputs of the node-level attention layer are the nodes initial feature vectors ${h}$ ($h_{n_{i}}, h_{s_{i}}, h_{c_{i}}$)
    * Since we have **multiple types of nodes**, we will naturally have our **initial feature vectors** belonging to **feature spaces of different dimensions**. 
    $\rightarrow$So in order for the attention to output comparable and meaningful weights between these types, we ought use a **type-specific transformation matrix** to **project features** with different dimensions **into the same feature space**.
    * In the ~={cyan}context of Fake news detection=~, The **~={cyan}target node=~** is the **~={cyan}news article node=~** $n_{i} \in \mathcal{N}$. 
    * Its **neighbours belong** to $\mathcal{N \cup S\cup C}$ . The **target node** is itself **considered a neighbour** to cooperate the attention mechanism. 
    * Let $T\in\{\mathcal{N,C,S}\}$ and nodes in $T$ have the type $\phi_{t}$.
    * The **projection operation** would look like:
    $$h'_{t_{j}} = \mathbf{M^{\phi_{t}}}.h_{t_{j}}$$
    * With: $M^{\phi_{t}} \in \mathbb{R}^{F\times F^{\phi_{}}}$ : the transformation matrix of type $\phi_{t}$
    $\rightarrow$ $F^{\phi_{t}}$ is the dimension of the initial feature $h_{t_{j}} \in \mathbb{R}^{\phi_{t}}$ of the node $t_{j}$
    $\rightarrow$ $F$ is the dimension of the feature space mapped to, being the **same** for all type-specific transformation matrices.
    $\rightarrow h'_{t_{j}}$ is the projected feature of node $t_{j}$ 
    
* Now, for $n_{i}$'s neighbour nodes in $T$, **The node-level attention** can learn the **importance** $e^{\phi_{t}}_{ij} \rightarrow$ ~={cyan}*how important neighbour node $t_{j}\in T$ will be for target node $n_{i}$=~.*
    $$e^{\phi_{t}}_{ij}=att(h'_{n_{i}},h'_{t_{j}};\phi_{t})$$ $\rightarrow att$ is shared across all same-type $\phi_{t}$ neighbours nodes.
$\rightarrow$ **The masked attention** captures the network structure so that **only node $t_{j}\in neighbour_{n_{i}}$ is calculated and recorded as $e^{\phi_{t}}_{ij}$**, otherwise the attention weight will be zero.
* We then **normalize** the recorded attention weights using ***softmax function*** to get the ~={cyan}weight coefficient=~ $\alpha^{\phi_{t}}_{ij}$
$$\alpha^{\phi_{t}}_{ij}=softmax(e^{\phi_{t}}_{ij})=\frac{\exp(e^{\phi_{t}}_{ij})}{\Sigma_{t_{k}\in neighbour_{n_{i}}}\:\exp(e^{\phi_{t}}_{ik})} $$
Now we **aggregate** our schema node $T_{n_{i}}$ by **neighbour projected features with the corresponding weights**
$$T_{n_{i}}=\sigma(\Sigma_{t_{j}\in neighbour_{n_{i}}}\; \alpha^{\phi_{t}}_{ij}.h'_{t_{j}})$$ 
*  Similar to Graph Attention Network (GAT), we can use **Multi-head attention mechanism** to stabilize the learning process of self-attention in node-level attention.
$\rightarrow$ $K$ independent node-level attention execute the transformation of the schema node equation above. 
$\rightarrow$The features achieved by $K$ heads will be then concatenated, resulting in the following output representation of the schema node:
$$\parallel^{K}_{k=1} T_{n_{i}}=\sigma(\Sigma_{t_{j}\in neighbour_{n_{i}}}\;\alpha^{\phi_{t}}_{ij}.h'_{t_{j}})$$
with: $\parallel$ denoting concatenation.


 2. ~={cyan}**Schema-level attention**:=~ 
  * *Through node-level attention, we fuse information from same-type neighbour nodes into the representation of a schema node.* 
  * For each target node $n_{i}$, we have **three schema nodes** denoted as $\mathcal{N_{n_{i}}, C_{n_{i}},S_{n_{i}}}$, we learn their importance and use the learned coefficients for weighted combination.
  * In order to compute the attention weights, first, we apply a ~={cyan}**linear transformation** to the schema nodes=~, **parametrized by a weight matrix** $\mathbf{W}\in\mathbb{R}^{F'\times KF}$, where $K$ is the number of heads in node-level attention.
  * The **schema-level attention** $schema$ is a **1-layer FFN** that applies **sigmoid function** with dimension of $2F'$ .
  * The~={cyan} importance of the schema node=~ $T_{n_{i}}$ is denoted as $w^{\phi_{t}}_{i}$: $$ w^{\phi_{t}}_{i} = schema(\mathbf{W}T_{n_{i}},\mathbf{W}\mathcal{N}_{n_{i}} ) $$
   * We then normalize the importance of each schema node with *softmax function* to get the ~={cyan}coefficients of the final fusion=~ $\beta^{\phi_{t}}_{i}$ $$\beta^{\phi_{t}}_{i}=softmax(w^{\phi_{t}}_{i})=\frac{\exp(w^{\phi_{t}}_{i})}{\Sigma_{\phi\in \mathcal{V}_{T}}\:\exp(w^{\phi}_{i})}$$
  * Based on the learned coefficients, we fuse all schema node to get the ~={cyan}**final representation**=~ $r_{n_{i}}\in\mathbb{R}^{F'}$ of the target node $n_{i}$: $$ r_{n_{i}}= \Sigma_{\phi_{t}\in\mathcal{V}_{T}}\;\beta^{\phi_{t}}_{i}.T_{n_{i}}$$


#### ~={blue} **C) HGAT-based classifier explained**=~
![[Pasted image 20251223140514.png]]
* The **inputs** of HGAT-based classifier are the **initial feature vectors of nodes** $\{h\}$.
* The classification layer **outputs the predicted labels $\{\hat{y}\}$ of unlabelled news article nodes**. *Logistic regression* works just fine in here.
* For FND, **optimizing HGAT-based classifier** can be done via backpropagation and leveraging the **cross-entropy loss minimization**. 
* Given the set of labelled news article nodes $\mathcal{N}_{L}$, and the unlabelled set $\mathcal{N}_{U}$, the cross entropy loss can be written as:
 $$ Loss_{classifier}= - \Sigma_{n_{i}\in\mathcal{N}_{L}} \;(y_{n_{i}}\log(p_{n_{i}})+(1-y_{n_{i}})\log(1-p_{n_{i}}))$$
* Where: $p_{n_{i}}$ is ~={cyan}the predicted probability of labelled news article node=~ $n_{i}$
        $y_{n_{i}}$ is a ~={cyan}binary indicator=~ indicating if the ~={cyan}binary class label=~ (Real, Fake) is the ~={cyan}correct classification=~ for the new article node representation $r_{n_{i}}$

* Optimization completed, The **predicted probabilities of unlabelled news article nodes in** $\mathcal{N}_{U}$ are rounded and cast into **predicted labels** $\{\hat{y}\}$ to be ~={cyan}**evaluated by HGAT-based selector**=~.


#### ~={blue} **D) HGAT-based selector explained**=~
![[Pasted image 20251223142708.png]]
* The **inputs** of HGAT-based selector are the **initial feature vectors of nodes** $\{h\}$
* Based on the final learned representation $r_{n_{i}}$, we **concatenate $r_{n_{i}}$ with the predicted label $\hat{y}$ of the unlabelled node (or the ground-truth $y$ of the labelled node)** to get the **vector** $z_{n_{i}}\in\mathbb{R}^{(F'+1)}$ $$ z_{n_{i}}=[r_{n_{i}},\hat{y}]$$
* The ~={cyan}purpose of HGAT-based selector=~ is to **evaluate the probability that how likely the $z_{n_{i}}$ is from the set** $\mathcal{N}_{L}$ (labelled news article nodes). 
    * Higher possibility means a news article node $n_{i}\in \mathcal{N}_{L}$ **matches** the predicted label better.
    * In the **absence of a match**, it likely means that the **predicted label is wrong**.
* The output layer serves to **predict the probability/confidence** $\mathcal{P}(\hat{y};r_{n_{i}})$. *Logistic regression* works just fine.
* $z_{n_{j}}$, with  $n_{j} \in \mathcal{N}_{L}$ are sampled as **positive samples**, and the same number of $z_{n_{k}}$ with $n_{k}\in \mathcal{N}_{U}$ are sampled as **negative examples**. These two samples constitute the **training set** for the HGAT-based selector.
* The loss function is a **cross-entropy loss** that can be optimized via backprop:  $$ Loss_{selector}= - \Sigma\;(y\log(\mathcal{P})+(1-y)\log(1-\mathcal{P}))$$ with:  $y\in\{0,1\}$ is the negative-positive label of the concatenated vector in training set.
        $\mathcal{P}$ is the predicted probability of label being positive. 
* The **rest of concatenated vectors of unlabelled news article node** are the **test set**. 
* After training, HGAT-based selector outputs the probability $\mathcal{P}$ for test set.
* Based on the probability, we propose a **query strategy** to select **high-value candidates for active learning, aka, to be labelled by experts.** 
    * ~={cyan}Lower=~  $\mathcal{P} \rightarrow$ ~={cyan}**unlabelled news article node**=~ and ~={cyan}**predicted label=~ don't match**.=~
             $\rightarrow$ ~={cyan}High probability that the predicted label given by the HGAT-based classifier is **wrong**.=~
             $\rightarrow$ The wrongfully classified node can be more **informative** and become **part of the training set** in the next round of training **after experts labelling**.
    * **Query strategy** $\rightarrow$ All samples from test set will be sorted in ascending order according to to the predicted probability $\mathcal{P}$. 
      The top $k$ candidates will be added to the query set $U_{q}$, $k$ is the query batch size.

#### ~={blue} **E) Adversarial Active Optimization**=~
![[Pasted image 20251223165148.png]]
* ~={yellow}HGAT-based classifier=~ and ~={yellow}HGAT-based selector=~ cooperate in **an adversarial active manner**.
* In each iteration, **HGAT-based classifier** and **selector** are trained ~={yellow}*alternately (one after the other)*=~
    * First, HGAT-based classifier is trained to output the predicted labels.
    * Then, HGAT-based selector is trained by these predicted labels from the classifier.
    * Based on the optimized selector, $k$ candidates will be queried in one iteration and will be added to $\mathcal{U}_{q}$ to be used as training data in the next iteration.
    * With **each $k$ candidates obtained**, the **performance of the classifier** is **improved** in the next iteration. $\rightarrow$ *~={yellow}**Better predicted labels by the classifier improves the evaluation performance of the selector**=~*.
* This above iteration is repeated until the size of $\mathcal{U}_{q}$ exceeds the query budget $b$.
