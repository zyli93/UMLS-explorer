from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import clean

POINCARE_DIR = "../../../poincare/icd10/icd10.pth.best"
MRCONSO_DIR = '../../2018AB_RRF/META/MRCONSO.RRF'

class PoincareDataset:
    """ Poincare dataset

    Attributes:
        embeds: pretrained poincare embedding
        id2phrase: defaultdict mapping id to phrase
        id2words: defaultdict mapping id to cleaned words
        phrase2id: defaultdict mapping phrase to id
        vocab: defaultdict mapping word to a list of phrase ids
    """
    def __init__(self, embeds, aui_str_pairs):
        self.embeds = nn.Embedding.from_pretrained(embeds)
        self.id2phrase = defaultdict()
        self.id2words = defaultdict(list)
        self.phrase2id = defaultdict()
        self.vocab = defaultdict(list)

        for i, (aui, phrase) in enumerate(aui_str_pairs):
            if phrase:
                self.id2phrase[i] = phrase
                self.phrase2id[phrase] = i
                lemmas = clean.clean_line(phrase)
                self.id2words[i] = lemmas
                for l in lemmas:
                    self.vocab[l].append(i)

    def lookup(self, word):
        return [(self.id2words[i], self.embeds(torch.LongTensor([i]))) for i in self.vocab[word]]
    
def export_embeddings(model):
    """Export pretrained embeddings from poincare model
    Args:
        model: pytorch poincare model
    Returns:
        a Tensor of embeddings
    """
    print("STATUS: exporting embeddings.tsv")
    np.savetxt("embeddings.tsv", model['embeddings'].numpy(), delimiter='\t')
    return model['embeddings']

def export_labels(model):
    """Export pretrained embedding labels from poincare model
    Args:
        model: pytorch poincare model
    Returns:
        a list of (AUI, STR) tuples
    """
    # import MRCONSO to map AUIs to STRs
    print("STATUS: importing MRCONSO.RRF")
    mrconso = pd.read_csv(MRCONSO_DIR, sep='|', header=None, dtype=object)
    mrconso = mrconso.drop(18, axis=1) # last column is meaningless because the entry ends with '|'
    mrconso.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"]
    assert(not mrconso['AUI'].duplicated().any())
    aui2str = dict(zip(mrconso['AUI'], mrconso['STR']))

    aui_str_pairs = list()
    # export labels tsv
    print("STATUS: exporting labels.tsv")
    with open("labels.tsv", "w") as fout:
        print('AUI\tString', file=fout)
        for aui in model['objects']:
            if aui in aui2str:
                phrase = aui2str[aui]
                aui_str_pairs.append((aui, phrase))
                print(aui, end='\t', file=fout)
                print(phrase, file=fout)
            else:
                aui_str_pairs.append((aui,None))
                print(aui, file=fout)
                print("WARNING: AUI {} not in MRCONSO".format(aui))
    return aui_str_pairs

def load_pretrained(dir):
    """Load pretrained poincare model into PoincareDataset
    Args:
        dir: pytorch poincare model directory
    Returns:
        a PoincareDataset object
    """
    print("========================Loading Pretrained Poincare Model========================")
    model = torch.load(dir)
    embeds = export_embeddings(model)
    aui_str_pairs = export_labels(model)
    dataset = PoincareDataset(embeds, aui_str_pairs)
    print("Summary:")
    print("\tPhrase/embedding size: {}".format(len(dataset.phrase2id)))
    print("\tWord/vocabulary size: {}".format(len(dataset.vocab)))
    print("=====================Finish loading Pretrained Poincare Model=====================")
    return dataset

if __name__ == "__main__":
    dataset = load_pretrained(POINCARE_DIR)