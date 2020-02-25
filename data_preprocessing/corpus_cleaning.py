"""
    Clean medline and pmc corpora:
    lowercase all words,
    remove punctuations, contractions, stop words,
    keep compound words,
    lemmatize all words,
    save frequency of words
    
    Author: Louis Qin <louisqin@ucla.edu> or <qyl0509@icloud.com>

    NLTK reference:
    1. https://medium.com/@pemagrg/nlp-for-beginners-using-nltk-f58ec22005cd
    2. https://machinelearningmastery.com/clean-text-machine-learning-python/
"""

import os
import re
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

MEDLINE_SRC_DIR = "../../medline_output"
MEDLINE_DEST_DIR = "../../medline_clean"

def sentence_cleaning(sentence):
    """return a cleaned list of words from a sentence"""
    # lowercase sentences
    lowered = sentence.lower()
    # tokenize word
    words = word_tokenize(lowered)
    # remove punctuation and stop words
    cleaned = [w for w in words if re.search('[a-zA-Z]', w) and not w in stopwords.words('english')]
    # lemmatize words
    wl = WordNetLemmatizer()
    lemmas = [wl.lemmatize(w) for w in cleaned]
    return lemmas

def medline_cleaning():
    """clean medline corpus"""
    freq = FreqDist([])
    with os.scandir(MEDLINE_SRC_DIR) as files:
        for file in files:
            src_dir = os.path.join(MEDLINE_SRC_DIR, file.name)
            dest_dir = os.path.join(MEDLINE_DEST_DIR, file.name)
            print("processing {}...".format(src_dir))
            with open(src_dir, 'r') as f:
                paragraph = f.read()
            sentences = sent_tokenize(paragraph)
            for s in sentences:
                lemmas = sentence_cleaning(s)
                # write cleaned sentences
                with open(dest_dir, 'a+') as fout:
                    print(' '.join(lemmas), file=fout)
                # update word frequency
                freq += FreqDist(lemmas)
    with open('medline_frequency.pickle', 'wb') as handle:
        pickle.dump(freq, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pmc_cleaning():
    """clean pmc corpus"""
    # TODO: handle pmc format
    return

if __name__ == "__main__":
    medline_cleaning()