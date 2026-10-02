
- **Core Problem & Motivation:** Existing GNN-based fake news detectors predominantly treat detection as binary classification and rely on static, homogeneous neighborhood aggregation. Uniform message passing across fixed graph topologies often introduces over-squashing and over-smoothing, while ignoring speaker context and multi-class veracity gradations.
    

### ~={blue}Proposed Architecture: DHGAT=~

- **Heterogeneous Graph Construction:**
    - News items constitute graph nodes, while contextual metadata (such as speaker profile attributes, subject matter, and affiliations) define multiple edge relation types ($R$) connecting news nodes.
    - Node text representations are initialized using FastText subword embeddings to robustly capture rare tokens and morphological variations.
- **Decision Network ($\Phi$):**
    - Employs a learnable categorical gating layer powered by the Gumbel-Softmax distribution.
    - Allows each individual node to differentiably and dynamically select its optimal neighborhood edge type at each layer, tailoring message aggregation to its specific contextual requirements.
- **Representation Network ($\Psi$):**
    - Applies heterogeneous Graph Attention Networks (GAT) over the dynamically selected neighborhood to update node embeddings through masked self-attention.
    - Mitigates over-squashing by filtering uninformative or noisy propagation pathways.
- **Semantic Distance-Aware Loss:**
    - Implements an MLP classifier optimized via a custom loss function over multi-class outputs.
    - Penalizes classification errors proportionally based on the semantic distance between predicted and true ordinal truth ratings (e.g., distinguishing subtle spin from outright fabrication).

### ~={blue}Benchmark Evaluation & Key Findings=~

- **Dataset & Setup:** Evaluated on the 6-class LIAR benchmark (12,836 statements) under semi-supervised settings with limited labeled data (10%, 20%, and 30% label availability).
- **Performance:** Improves multi-class classification accuracy by approximately 4% over competing baselines, maintaining strong performance even under constrained supervision.