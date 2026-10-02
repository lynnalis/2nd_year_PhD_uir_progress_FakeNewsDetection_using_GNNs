[MuMiN](https://dl.acm.org/doi/epdf/10.1145/3477495.3531744) is a massive and ~={cyan}multilingual misinformation graph dataset=~ that contains ~={cyan}rich social media data=~ *(tweets, replies, users, images, articles, hashtags)* spanning **21 million tweets** ***belonging to 26k twitter threads***, each to have been *~={cyan}semantically linked to 12k fact-checked claims=~* across many *topics, events and domains*, in **41 different languages**, spanning more than a **decade**.
==> Here is the link to their github repository: [MuMiN-dataset/mumin-trawl: The source code used to construct the MuMiN dataset from scratch.](https://github.com/MuMiN-dataset/mumin-trawl) 
==>The information was organized into a **heterogeneous graph**.

#### ~={pink}**Dataset Creation**=~
It consists of *two parts*:
* Collection of claims and their fact-checked verdicts.
* Collection of the surrounding social context.

==***Claims:***==
* In terms of content and language diversity, The **fact-checked claims** are collected using [^1~={yellow}]**Google Fact Check Tools API**=~. Based on keywords 'coronavirus' and 'covid', the search resulted in **115 active fact-checking organizations**, from which they ~={yellow}scrap all historical fact-checked claims=~ to the present day (theirs not ours), totalling in **128070 claims**.
* For each claim, we extract ~={yellow}key metadata=~, including *source, reviewer URL, language, fact-checking verdict, date*.
* As for the *verdict* being unstructured free text in various languages, after manual labelling of 2500 unique verdicts, we train a **verdict classifier** to classify the free-text verdicts into 3 pre-specified categories:
    - **misinformation:** for false, misleading, half true, half false.
    - **factual:** for true, correct, mostly true.
    - **other:** for when it's not clear (the verdict only discusses the claim without assessing its veracity) (verdicts with this label are later dropped from the final dataset)
* ~={red}Note:=~ 
    * *The verdicts were all translated to English and a roberta-base model was trained for the verdict classification.*
    * *The authors tried to work with a multilingual model (xlm-roberta-base) for the classification after translating the 2500 labelled verdicts to 65 languages, but that approach wasn't as good (for the low-resource languages).*
![[Pasted image 20260313113240.png]] 
==***Twitter***==
* Now comes the collection of **social media context**, precisely, from twitter using **Twitter Academic Search API** to get as ~={yellow}many relevant Twitter Threads sharing and discussing content related to the claims.=~
* We use *KeyBERT and a Sentence Transformer* to extract **top 5 keyphrases** from each fact-checked claim, to be fed to the search API and fetch for 100 results per keyphrase *(non-reply tweets with a link or image, posted no more than 3 days prior to the claim date)*, resulting in **30 million tweets**.
* We then filter and keep the ~={yellow}viral tweets=~ *(tweets with a minimum of 5 retweets)*, which kept **2.5 million tweets**. 
* We ~={yellow}extract all URLs and hashtag=~s. If a URL leads to an article, we download the *title, body, top image* using *newspaper3k*. We also ~={yellow}extract hyperlink for all shared images=~.  
* All the acquired information was populated in a **graph dataset** with about **17 Million nodes** and **50 million relations**, using *Neoj4 framework. ~={yellow}All nodes are unique=~. 

==***Data Linking***==
* In order to match multilingual tweets, claims and articles, we **translate all text to English** using ~={yellow}Google Translation API=~. (there's the risk of losing context)
* We then **summarize the concatenated title and abstract** of collected articles to fit within the token limits of the embedding models.
* We then **embed the translated claims, tweets, and summarized articles** into *vector representations* using a Sentence Transformer *(paraphrase-mpnet-base-v2)*
* In order to **link relevant tweets to the claims**, we group the claims in batches of 100, then compute the ~={yellow}cosine similarity=~ between them in a 3-days window. 
* Based on a qualitative evaluation of these similarities, we produce ~={yellow}3 datasets=~ *(small, medium, large)* using strict similarity thresholds of **0.8, 0.75, 0.7** respectively.
![[Pasted image 20260313113322.png]]
==***Data Enrichment***==
* After linking the twitter posts to the claims, ~={yellow}we query Twitter API for the surrounding context of these posts=~.
* For each tweet, the context include:
    - 100 users who retweeted.
    - 100 followers of the tweet's author.
    - 100 users followed by the author.
    - 500 users who replied.
    - All users mentioned within the tweet along with their recent 100 tweets.


==>Down below is the graph schema of the dataset:
![[Pasted image 20260313115524.png]]

==***Limitations***==
* Given the automated linking procedure between claims and tweets, claims and articles, wrongful labels could exist. (issue can be addressed by going with high similarity thresholds)
* In the absence of judgements towards impartiality or correctness of the verdicts provided by the fact-checking organizations, there could be biased or contentious or inaccurate verdicts in the dataset.
* The classification of the free-text verdicts is done with a machine learning model, and in spite of its high performance of a test set, there's the possibility of misclassifying some verdicts. 
* Last but not least, conducting a reproducible research can be challenging. In fact, the raw social network data is not handed but instead, it's the code to retrieve it that's provided, so if a user deletes their tweet or account, retrieving their data from Twitter API is impossible.

[^1]: It's a resource hat collects fact-checked claims from fact-checking organisations around the world.
