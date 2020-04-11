# Evaluation of Medical Word Embedding

Here are some ideas of embedding evaluation from the three papers.

## Wang et al. (2018) -- Mayo Clinic

A comparison of word embeddings for the biomedical natural language processing.

### Comparison

**Between** embeddings trained from new algorithm **and** embeddings of pure GloVe.

### Qualitative eval

#### Print similar words

Randomly select medical terms from three categories (i.e., disorder, symptom, and drug), and manually inspect the five most similar words computed by embeddings for each term.

Words from three categories: disorder, symptom, and drug and measure the similarity by

sim$(w_1, w_2) = \frac{\theta_1\cdot\theta_2}{\left\Vert\theta_1\right\Vert\left\Vert\theta_2\right\Vert}.$

If the term is a phrase with embedding $\Theta_1$, then the embedding is the average  of the all the words in it $\Theta_1=\frac{1}{n}\sum_{i=1}^n\theta_i$.

Three categories:

1. Disorder: diabetes; peptic ulcer disease; Colon cancer;
2. Sympton: dyspnea; sore throat; low blood pressure;
3. Drug: opioid; aspirin.



#### Visualize terms

Analyze the word embeddings through a 2-dimensional visualization plot of 377 medical terms. 377 medical terms are from **UMNSRS** dataset. And the visualization is done by t-SNE.

### Quantitative eval

#### Intrinsic evaluation

evaluate the word embeddings’ ability to capture medical semantics by measruing the semantic similarity between medical terms.

1. **Pederson's dataset: 30 terms**
2. **Hliaoutakis's dataset: 34 terms**
3. **MayoSRS: 101 terms (need to normalize score)**
4. **UMNSRS: 566 terms**

#### Extrinsic evaluation

apply word embeddings to multiple downstream biomedical NLP applications with data from shared tasks.

1. clinical information extraction (IE)

   1. Local institutional dataset (unavailable)

      - Binary classification
      - 10-Fold Cross Validation
      - Avg word embedding as string embedding
      - SVM

   2. **i2b2 2006 smoking status extraction**

      Identical settings as above.

2. biomedical information retrieval (IR)

   * **TREC 2016 CDS Track**
   *  (The requirement is too hard to understand. TODO)

3. relation extraction (RE)

   * **DDIExtraction 2013 Corpus**
   * Method in here: *Dependency embeddings and amr embeddings for drug-drug interaction extraction from bio- medical texts*.
   * Random Forest, 10-Fold CV, detailed method in page 18.

##### Evaluation metrics

- Pearson's correlation + t-test of statistical significance

#### OOV problem

A method like fastText is used. Details are no page 16.

### Other resources:

pubmed stopwords list:  http://www.ncbi.nlm.nih.gov/books/NBK3827/table/pubmedhelp.T.stopwords/.





## Khattak et. al. 2019

We only focus on the evaluation part.

### Intrinsic

#### For general word embeddings

##### Method 1

* SimLex-999 [63]  -- 999 word pairs

* WordSim353 [64] -- 353 word pairs
* MEN [65] -- 3000 word pairs

##### Method 2

measure word embedding by learning an alignment between the word embeddings and manually-constructed feature vectors with linguistics properties. Code is [here](https://github.com/ytsvetko/qvec)

#### For medical embeddings

##### evaluation methods

- UMNSRS/MayoSRS
- Medical Conceptual Similarity Measure (MCSM)/Medical Relatedness Measure (**MRM**); *Learning Low-Dimensional Representations of Medical Concepts*

#####  evaluation for word embeddings for clinical text

Chiu et al. :

- Intrinsic: UMNSRS-Rel/UMNSRS-Sim
- Extrinsic:
  - NER on BioCreative II Gene Mention task (BC2)
  - JNLPBA corpus (PBA)

Want et al.: 略

Huang et al:

* elementary vector representations (EVR)
* Other skipped

De Vine et al.

* Clinical information extraction
* CRF on two clinical corpora:
  * Concatenation of i2b2 trainset, MedTrack, CLEF 2013 train and test sets
  * i2b2 train set only and two non-clinical corpora -- PubMed and Wikipedia

##### evaluation of clinical concept embeddings

1. MaysoSRS + UMNSRS
2. **Zhao et al. [43] DrugBank.** coherence assessment and outlier detection
3. Yu et al. [Missing Citation in paper]. **metric** to determine the degree of semantic similarity between pairs of concepts.
4. ClinicalBERT has **another dataset [51]**



### Extrinsic

1. **Chiu et al. [66]** relationship between intrinsic and extrinsic. 10 benchmark datasets.
2. For extrinsic, three tasks: part-of-speech tagging, chunking, and named entity recognition. "*While good scores on intrinsic evaluation tasks may demonstrate that the embed- dings are capturing coherent information, it may not be useful for downstream tasks.*"
3. Si et al [80] identifying clinical concepts in text
   1. i2b2 2010/2012 (clinical notes with annotated concepts)
   2. clinical reports with disease concepts from SemEval 2014/2015
   3. Other resource MIMIC-III
4. Predicting unplanned readmission after discharge
   1. Craig et al. [37] and Nguyen et al. [38] use CNN
   2. Pham et al. [39] uses an RNN
5. ICD code prediction. Patel et al. [35] and Escudie et al. [33]
6. Patient Phenotyping: Gehrmann et al. [36]
7. Other clinical predictive tasks: [29], [47], [30], [32], [34]
8. Named Entity Recognition 
   1. [61] i2b2 dataset
   2. [43] drug name recognition and classification
   3. [17] Hospital readmission within 30 days. MIMIC-III
   4. [16] two NER, two de-identification, and NLI
   5. [19] SciBERT: NER, PICO (participant intervention comparison outcome extraction), classification, dependency parsing



## Shulz et al. 2020

We present multiple automatically created large-scale medical term similarity datasets and confirm their high quality in an annotation study with doctors.

### Intro

Only based on terms!

### Related work

Half contain only single-word terms, although medical terms are frequently made of **multiple words**. Note that we here focus on medical *terms* rather than *concepts*. Concepts are abstract entities, represented as codes in ontologies such as SNOMED, which are described by some (potentially more than one) term.

### New Large-Scale Datasets

* Basis: **SNOMET Clinical Terms (CT)** most comprehensive, multilingual clinical healthcare terminology in the world.

### Methods for Measuring Term Similarity

#### Word and Contextual Embeddings

- word2vec skip-gram
- fastText
- Non-medical: GloVe and Fasttext on Wikipedia and Common Crawl
- Contextual embeddings: ELMo/ELMoPubMed/Flair/SciBERT/GPT

#### Similarity Metrics

- Single word term: word vector

- Multi word term: averaging and summing
- Rank correlation coefficients Pearson's, Spearman's, and Kendall's by Zhelezniak et al. (2019a).
- fuzzy Jaccard and max Jaccard similarity for multi-word strings. Zhelezniak (2019b).





