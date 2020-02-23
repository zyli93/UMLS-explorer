# UMLS Explorer

## Table of contents
- [MRHIER](##MRHIER)
- [NET](##NET)
- [data_preprocessing](#data_preprocessing)

## MRHIER
The files under this directory explores the Computable Hierarchy MRHIER.RRF within UMLS Metathesaurus. \
For more information about MRHIER.RRF, reference [UMLS Reference Manual 3.3.11](https://www.ncbi.nlm.nih.gov/books/NBK9685/#ch03.sec3.11)

## NET
This directory includes an interactive visualization script to play UMLS Semantic Network data.

## data_preprocessing
`transitive_closure.py` creates transitive closure for UMLS source vocabulary (currently only ICD10) for poincare embedding model. \
`corpus_token.py` tokenizes corpora and applies stemming and lemmatization.
