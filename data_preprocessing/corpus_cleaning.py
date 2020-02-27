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
import multiprocessing
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

MEDLINE_SRC_DIR = "../../medline_output"
MEDLINE_DEST_DIR = "../../medline_clean"
MEDLINE_FREQ_DIR = "../../medline_freq"
PMC_SRC_DIR = "../../pmc_output"
PMC_DEST_DIR = "../../pmc_clean"
PMC_FREQ_DIR = "../../pmc_freq"

def file_cleaning(src_dir, dest_dir, freq_dir):
    """clean a file and count clean words frequencies"""
    print(f"processing {src_dir}...")
    freq = FreqDist([])
    with open(src_dir, 'r') as f:
        paragraph = f.read()
    sentences = sent_tokenize(paragraph)
    for s in sentences:
        # Special lines in pmc
        if s.startswith(">>>>>>"):
            s = s.split(None, 1)[1]
        # lowercase sentences
        lowered = s.lower()
        # tokenize word
        words = word_tokenize(lowered)
        # remove punctuation and stop words
        cleaned = [w for w in words if re.search('[a-zA-Z]', w) and not w in stopwords.words('english')]
        # lemmatize words
        wl = WordNetLemmatizer()
        lemmas = [wl.lemmatize(w) for w in cleaned]
        # write cleaned sentences
        with open(dest_dir, 'a+') as fout:
            print(' '.join(lemmas), file=fout)
        # update word frequency
        freq += FreqDist(lemmas)
    with open(f'{freq_dir}.pickle', 'wb') as handle:
        pickle.dump(freq, handle, protocol=pickle.HIGHEST_PROTOCOL)

def medline_cleaning():
    """clean medline corpus"""
    with os.scandir(MEDLINE_SRC_DIR) as files:
        dir_args = [(os.path.join(MEDLINE_SRC_DIR, file.name), \
                    os.path.join(MEDLINE_DEST_DIR, file.name), \
                    os.path.join(MEDLINE_FREQ_DIR, file.name)) for file in files]
    with multiprocessing.Pool() as pool:
        pool.starmap(file_cleaning, dir_args)

def pmc_cleaning():
    """clean pmc corpus"""
    with os.scandir(PMC_SRC_DIR) as files:
        dir_args = [(os.path.join(PMC_SRC_DIR, file.name), \
                    os.path.join(PMC_DEST_DIR, file.name), \
                    os.path.join(PMC_FREQ_DIR, file.name)) for file in files]
    with multiprocessing.Pool() as pool:
        pool.starmap(file_cleaning, dir_args)

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=medline_cleaning)
    p1.start()
    p2 = multiprocessing.Process(target=pmc_cleaning)
    p2.start()
    p1.join()
    p2.join()