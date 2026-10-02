
#### **==Terminology:==**
##### **Defining News-HIN:**
* News-HIN can be defined as $\mathcal{G}=(\mathcal{V,E})$
    * Node set $\mathcal{V=C\cup N \cup S}$  with
        * $\mathcal{C}$: creators denote people who write news articles.
        * $\mathcal{N}$: News articles refer to news content post on social media o public platforms.
        * $\mathcal{S}$: Subjects are the central ideas of news articles. 
    * Link set $\mathcal{E=E_{c,n} \cup E_{n,s}}$ with $\mathcal{E_{c,n}}$ referring to *Write* link between creators and news articles, and $\mathcal{E_{n,s}}$ referring to *Belongs to* link between news articles and subjects. 
##### **Defining News articles:**
* The News articles set is represented as $\mathcal{N}=\{n_{1},n_{2},\dots,n_{m}\}$. 
* For **each news article** $n_{i}$, we have its **textual contents**. 
* The credibility label of $n_{i}$ takes value from the label set $\mathcal{Y}=\{Fake,Real\}$.
* Originally, there were 6 labels partitioned like so: $Fake$={Pants on Fire, False, Mostly False} and $True$={True, Mostly True, Half True}. 
##### **Defining Subjects:**
* The Subjects set is represented as : $\mathcal{S}=\{s_{1},s_{2},\dots, s_{n}\}$.
* For **each subject** $s_{i} \in \mathcal{S}$, we have its **textual description**.
##### **Defining Creators:**
* The Creators set is represented as : $\mathcal{C}=\{c_{1},c_{2},\dots, c_{n}\}$.
* For **each creator** $c_{i} \in \mathcal{C}$, we have its **profile information** (sequence of words). In [PolitiFact dataset](https://figshare.com/articles/dataset/PolitiFact_dataset/28614938)), this information includes the titles, the political party membership and the geographical residential locations.

Defining the **schema-level description** is important to better understand News-HIN and learn the importance of nodes and links with different types.
##### **Defining News-HIN schema:**
* The  Schema of News-HIN $\mathcal{G}=(\mathcal{V,E})$ can be defined as $\mathcal{S_{G}}=(\mathcal{V}_{T},\mathcal{E}_{T})$
    * Node types set $\mathcal{V}_{T}=\{\mathcal{\phi_{n},\phi_{c},\phi_{s}}\}$
    * Link types set $\mathcal{E}_{T}$=*{Write, Belongs to}*

#### **==Problem definition:
* Given a News-HIN $\mathcal{G}=(\mathcal{V,E})$, FND problem aims at ~={yellow}learning a classification function =~$f: \mathcal{N} \rightarrow \mathcal{Y}$ to ~={yellow}classify news article nodes=~ in $\mathcal{N}$ into the ~={yellow}correct class with the credibility label=~ in $\mathcal{Y}$.
* The news article nodes with labels are grouped as a **labelled set** $\mathcal{L}$.
* The rest of news article nodes is grouped in the **unlabelled set** $\mathcal{U}=\mathcal{N}\setminus\mathcal{L}$ 
* Based on the **active learning** settings, we are allowed to **query for labels of news article nodes in $\mathcal{U}$** with an upper limit $b$.
* We want to achieve an **optimal query set** $\mathcal{U_{q}}$ to **improve the classification function** $f: \mathcal{N} \rightarrow \mathcal{Y}$.
* To resolve the above fake news detection problem, we introduce the proposed **~={yellow}adversarial active learning based heterogeneous graph neural network**=~ AA-HGNN Next.