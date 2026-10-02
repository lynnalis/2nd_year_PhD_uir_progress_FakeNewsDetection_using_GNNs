
#### ~={green}  **Data description:**=~
![[Pasted image 20251223171622.png]]
* The ~={green}**main dataset**=~ is collected from the ~={green}**Fact-checking website *Politifact***=~, where **political statements or news articles (news article nodes)** *belonging to* **subjects (subject nodes)** based on their content and topics, are *written* by **politicians/political groups (creator nodes)**.
* The **fact-checking results (ground-truth labels)** provided by the website takes 6 values, which were partitioned in 2 groups for the experiments like so: $Fake$={Pants on Fire, False, Mostly False} and $True$={True, Mostly True, Half True}. There are **6465 fake news** and **7590 Real news**.
* The ~={green}**other dataset** *BuzzFeed*=~ is used to verify **Generalization and stability of AA-HGNN**, containing **91 fake news articles** and **91 Real news articles**.

#### ~={green}  **Experimental settings:**=~
* Since the target is to detect fake news, $Fake$ class is treated as **positive class** and $Real$ class is the **negative class**. 
* 20% of news article nodes is used as the training set, 10% for the validation set and the testing ratio is fixed at 10%. 
#### ~={green} **Data preprocessisng:**=~
* Both datasets has data with different length.
* We transform the input features of each type of nodes into a **vector with fixed length** using *TfidfVectorizer* to extract features. 
* For *Politifact dataset*, the dimensions of initial features: $dim(h_{n_{i}})=3000$, $dim(h_{c_{i}})=3109$,$dim(h_{s_{i}})=191$.
* For *BuzzFeed* dataset, the parameter for *max_features* set for $n_{i} \in \mathcal{N}$ is 3000.
#### ~={green} **Comparison methods:**=~
The framework **AA-HGNN** was compared to three categories:
![[Pasted image 20251224092910.png]]
#### ~={green} **Results:**=~
![[Pasted image 20251224093107.png]]
