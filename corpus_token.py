import os
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def stem(tokens):
    """stem a list of raw tokens"""
    root_tokens = []
    ps = PorterStemmer()
    for w in tokens:
        root_tokens += ps.stem(w)
    return root_tokens

def lemmatize(tokens):
    """lemmatize a list of raw tokens"""
    root_tokens = []
    wl = WordNetLemmatizer()
    for w in tokens:
        root_tokens += wl.lemmatize(w)
    return root_tokens

def pubmed_token():
    """tokenize pubmed corpus"""
    tokens = []
    path_to_medline = "../medline_output"
    with os.scandir(path_to_medline) as files:
        for file in files:
            with open(os.path.join(path_to_medline, file.name), 'r') as f:
                paragraph = f.read()
                tokens += word_tokenize(paragraph)
    return tokens

def pmc_token():
    """tokenize pmc corpus"""
    # TODO: handle pmc format
    return

if __name__ == "__main__":
    pubmed_raw_tokens = pubmed_token()
    with open('pubmed_raw_tokens.pickle', 'wb') as handle:
        pickle.dump(set(pubmed_raw_tokens), handle, protocol=pickle.HIGHEST_PROTOCOL)

    pubmed_stemmed_tokens = stem(pubmed_raw_tokens)
    with open('pubmed_stemmed_tokens.pickle', 'wb') as handle:
        pickle.dump(set(pubmed_stemmed_tokens), handle, protocol=pickle.HIGHEST_PROTOCOL)

    pubmed_lemmatized_tokens = lemmatize(pubmed_raw_tokens)
    with open('pubmed_lemmatized_tokens.pickle', 'wb') as handle:
        pickle.dump(set(pubmed_lemmatized_tokens), handle, protocol=pickle.HIGHEST_PROTOCOL)