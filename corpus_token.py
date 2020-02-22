import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

def stemming_lemmatisation():
    # TODO: implement stemming and lemmatisation
    return

def pubmed_token():
    tokens = []
    path_to_medline = "../medline_output"
    with os.scandir(path_to_medline) as files:
        for file in files:
            with open(os.path.join(path_to_medline, file.name), 'r') as f:
                paragraph = f.read()
                tokens += word_tokenize(paragraph)
    return tokens

def pmc_token():
    # TODO: handle pmc format
    return

if __name__ == "__main__":
    pubmed_tokens = set(pubmed_token())
    with open('pubmed_tokens.pickle', 'wb') as handle:
        pickle.dump(pubmed_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)