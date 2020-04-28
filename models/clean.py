"""
    Line cleaning:
        lowercase all words,
        remove punctuations, contractions, stop words,
        keep compound words,
        lemmatize all words,
    
    Author: Louis Qin <louisqin@ucla.edu> or <qyl0509@icloud.com>
"""

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_line(line):
    """Clean a line of words
    Args:
        line: a string of words
    Returns:
        a list of lemmas
    """
    # lowercase sentences
    lowered = line.lower()
    # tokenize word
    words = word_tokenize(lowered)
    # remove punctuation and stop words
    cleaned = [w for w in words if re.search('[a-zA-Z]', w) and not w in stopwords.words('english')]
    # lemmatize words
    wl = WordNetLemmatizer()
    return [wl.lemmatize(w) for w in cleaned]

if __name__ == "__main__":
    download_nltk_resources()