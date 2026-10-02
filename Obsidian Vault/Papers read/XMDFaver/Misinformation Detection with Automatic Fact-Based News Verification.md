
- **Core Problem & Motivation:** Fact verification benchmarks traditionally focus on verifying concise, isolated claims against ground-truth evidence, whereas online misinformation detection deals with lengthy, multi-faceted news articles. Direct adaptation is hindered by sequence length disparities and a lack of paired datasets containing full news articles alongside verified fact bases.
    

### ~={green}Proposed Architecture: XMDFaVer=~

- **Text Summarization Stage:**
    - Condenses long, noisy news articles into concise claim summaries while preserving key topical perspectives and core assertions.
- **Information Retrieval from Fact Pool:**
    - Utilizes a dedicated external fact pool containing verified justification articles.
    - Employs retrieval scoring to fetch the top-$k$ most relevant fact documents corresponding to the target claim.
- **Explainable Question Generation & Answering (Q&A):**
    - Generates clarifying questions from the synthesized claim and queries the retrieved fact articles to produce grounded answer spans.
    - Provides transparent, audit-ready verification pathways by making the evidence-checking process traceable and inspectable.
- **Authenticity Classification:**
    - Integrates the claim, retrieved evidence, and Q&A verification signals into a classifier to predict final news veracity.

### ~={green} Benchmark Datasets & Key Findings=~

- **Curated Datasets:**
    - **Extended-AveriTec:** Augments the AVeriTeC collection with full-text reference articles.
    - **Misbar Dataset:** Curates 4,051 expert-annotated news articles paired with justification documents from the Misbar fact-checking platform. Supports an 8-class fine-grained setup (_True, Fake, Misleading, Selective, Suspicious, Myth, Commotion, Satire_) and a standardized binary setup.
        
- **Empirical Outcomes:**
    - XMDFaVer consistently surpasses conventional content-based and fact-checking baselines across both benchmarks.
    - Verification accuracy scales positively with the number of retrieved fact articles, highlighting the value of evidence coverage.