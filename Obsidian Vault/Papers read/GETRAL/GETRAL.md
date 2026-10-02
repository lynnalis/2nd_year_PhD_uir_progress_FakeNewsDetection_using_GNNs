## ~={orange}Part I: Graph-Based Semantic Structure Mining Architecture
## =~
- **Core Problem & Motivation:** Sequential natural language models struggle with long-distance structural dependencies between dispersed claim snippets and evidentiary texts. Furthermore, representations derived from dense textual evidence often suffer from redundant noise and local evidence over-sensitivity.
#### Architectural Workflow & Graph Modeling:

- **Graph Construction:**
    - Plain text claims and evidentiary documents are converted into graph-structured representations using a fixed-size sliding window to define local neighborhood connectivity.
    - Co-occurring words within each window are connected by edges.
    - Identical words occurring across distant sentences are merged into a single node, pulling semantically linked snippets closer together to facilitate high-order message propagation across long textual distances.
    - Initial node features are assigned using pre-trained word embeddings.
- **Graph-Based Semantics Encoder:**
    - Utilizes gated graph neural networks (GGNN) across $T_E$ propagation layers to aggregate neighborhood information within a $T_E$-hop field.
- **Semantic Structure Refinement (SSR):**
    - Addresses informational redundancy in multi-document evidence by evaluating node relevance against both local context and claim semantics.
    - A fusion coefficient $\beta \in [0, 1]$ balances evidence-internal relevance against claim-directed relevance.
    - A discarding rate $r$ filters out low-scoring redundant nodes, adaptively attenuating irrelevant evidence while retaining salient clues.
- **Attentive Graph Readout Layer:**
    - Employs hierarchical pooling—comprising a word-level attentive layer and a document-level attentive layer—to synthesize refined node representations into claim-level and evidence-level semantic vectors for downstream interaction.
        

## ~={orange}Part II: Adversarial Contrastive Learning & Empirical Evaluation
## =~ ### Adversarial Contrastive Learning & Objective

- **Contrastive Learning Formulation:**
    - Implements a supervised contrastive loss ($\mathcal{L}_{cl}$) alongside standard cross-entropy classification ($\mathcal{L}_{ce}$) to maximize separability between authentic and deceptive news.
    - Pulls semantic representations belonging to the same veracity class close while pushing representations of opposing veracity classes apart.
    
- **Adversarial Gradient Perturbation:**
    - Generates worst-case feature perturbations along the gradient direction to create hard positive augmented instances without altering label semantics.
    - Enhances robustness against skewed class distributions, particularly on unbalanced datasets.
- **Joint Training Objective:**
    - Optimized via joint loss $\mathcal{L} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{cl}$, where $\lambda$ governs the auxiliary contrastive regularization weight.