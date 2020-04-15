from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

POINCARE_DIR = "../../../poincare/icd10/icd10.pth.best"
MRCONSO_DIR = '../../2018AB_RRF/META/MRCONSO.RRF'

class PoincareDataset:
    def __init__(self, embeds, aui_str_pairs):
        # Is torch nn.Embedding faster than tensor lookup
        self.embeds = embeds
        self.id2phrase = defaultdict()
        self.phrase2id = defaultdict()
        self.vocab = defaultdict(list)

        for i, (aui, phrase) in enumerate(aui_str_pairs):
            if phrase:
                self.id2phrase[i] = phrase
                self.phrase2id[phrase] = i
                # TODO: clean phrase
                for word in phrase.split():
                    self.vocab[word].append(i)

    def lookup(self, word):
        return [(id2phrase[i], embeds[i]) for i in self.vocab[word]]
    
def export_embeddings(model):
    print("exporting embeddings.tsv")
    np.savetxt("embeddings.tsv", model['embeddings'].numpy(), delimiter='\t')
    return model['embeddings']

def export_labels(model):
    # import MRCONSO to map AUIs to STRs
    print("importing MRCONSO.RRF")
    mrconso = pd.read_csv(MRCONSO_DIR, sep='|', header=None, dtype=object)
    mrconso = mrconso.drop(18, axis=1) # last column is meaningless because the entry ends with '|'
    mrconso.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"]
    assert(not mrconso['AUI'].duplicated().any())
    aui2str = dict(zip(mrconso['AUI'], mrconso['STR']))

    aui_str_pairs = list()
    # export labels tsv
    print("exporting labels.tsv")
    with open("labels.tsv", "w") as fout:
        print('AUI\tString', file=fout)
        for aui in tqdm(model['objects']):
            if aui in aui2str:
                phrase = aui2str[aui]
                aui_str_pairs.append((aui, phrase))
                print(aui, end='\t', file=fout)
                print(phrase, file=fout)
            else:
                aui_str_pairs.append((aui,None))
                print(aui, file=fout)
                tqdm.write("Warning: AUI not in MRCONSO {}".format(aui))
    return aui_str_pairs

if __name__ == "__main__":
    model = torch.load(POINCARE_DIR)
    embeds = export_embeddings(model)
    aui_str_pairs = export_labels(model)