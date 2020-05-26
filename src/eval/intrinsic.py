import numpy as np
from numpy.linalg import norm
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from nltk.stem import WordNetLemmatizer

UMLS_SIM_DIR = "../../evaluation/data/UMNSRS_similarity.csv"
UMLS_REL_DIR = "../../evaluation/data/UMNSRS_relatedness.csv"
TRAINED_EMB_DIR = "../models/corp2hype.pth"

def intrinsic_eval(model, lemma=False, debug=False):
    print("lemmatization", lemma)

    # TODO: verify glove (wi+wj) / 2
    embed = (model['glove']['wi.weight'] + model['glove']['wj.weight']) / 2
    word2id = model['word2id']

    sim_data = pd.read_csv(UMLS_SIM_DIR)
    rel_data = pd.read_csv(UMLS_REL_DIR)

    for data in (sim_data, rel_data):
        human_baseline = []
        cos_similarities = []
        terms_not_found = []
        lines_not_found = 0

        for _, row in data.iterrows():
            terms = [row['Term1'].lower(), row['Term2'].lower()]
            ids = []
            for i in range(2):
                if lemma:
                    wl = WordNetLemmatizer()
                    terms[i] = wl.lemmatize(terms[i])
                try:
                    id = word2id[terms[i]]
                    ids.append(id)
                except:
                    terms_not_found.append(terms[i])
        
            if len(ids) == 2:
                emb1, emb2 = embed[ids[0]].cpu().numpy(), embed[ids[1]].cpu().numpy()
                human_baseline.append(row['Mean'])
                cos_similarities.append(np.dot(emb1, emb2) / (norm(emb1) * norm(emb2)))
            else:
                lines_not_found += 1
        
        print("Pearson coefficient: ", pearsonr(cos_similarities, human_baseline))
        print("Spearman coefficient: ", spearmanr(cos_similarities, human_baseline))
        print(lines_not_found, len(data), "lines not found")
        if debug:
            print("Terms not found: ", terms_not_found)

if __name__ == "__main__":
    model = torch.load(TRAINED_EMB_DIR)
    intrinsic_eval(model)