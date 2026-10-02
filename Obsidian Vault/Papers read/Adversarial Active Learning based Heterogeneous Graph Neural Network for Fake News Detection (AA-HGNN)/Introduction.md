
* **On social media, news do not exist independently in the form of articles**, many entities such as ~={green}*news creators* (profile info is collected) =~and ~={green}*news subjects* (background and auxiliary knowledge is collected)=~ are related and can provide a more comprehensive perspective when identifying the credibility/ factuality/ truthfulness of news articles. 
* A **News Heterogeneous information network** or **News-HIN** is good to represent these entities and their relationships. Down below is an illustration of News-HIN according to *Politifact (a fact-checking website that rates the accuracy of claims by elected officials and others on its Truth-O-Meter.)*
![[Pasted image 20251222094251.png]]
* FND can be formulated as a **node classification** problem with the support of News-HIN.
* The news content and related entities are modelled as a News-HIN. 
* Both structural information and node content of News-HIN are utilized by AA-HGNN to identify fake news.
* This task comes with challenges which are:
    1. ~={green} *Scarcity of Training data*=~ : Fake news appear and spread quickly. Also, the real-time nature of news makes outdated labels useless. All resulting in a lack of valuable training data, which calls for training a FND model with small amount of training data. 
    2. ~={green} *Heterogeneity* =~: Learning effective node representations in a News-HIN considering both structural and type information (many types of heterogeneous info exist).
    3. ~={green}*Generalizability*=~ : The detection model needs to handle News-HINs having any types of nodes and different schemas.
