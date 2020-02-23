"""
    Create UMLS Source Vocabulary transitive closures for Poincare embedding.
    
    Author: Louis Qin <louisqin@ucla.edu> or <qyl0509@icloud.com>
"""

import pandas as pd

mrhier_filled = pd.read_csv('../MRHIER/MRHIER_filled.csv', sep='|', dtype=object)

source = mrhier_filled[mrhier_filled['SAB'] == 'ICD10']

edges = set()
for index, row in source.iterrows():
    if pd.isna(row['PTR']):
        continue
    parents = row['PTR'].split('.')
    for i, p in enumerate(parents):
        for c in parents[1:]:
            edges.add((c, p))
        edges.add((row['AUI'], p))

icd10 = pd.DataFrame(list(edges), columns=['id1', 'id2'])
icd10['weight'] = 1
icd10.to_csv('icd10_closure.csv', index=False)