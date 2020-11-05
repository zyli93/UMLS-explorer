"""
    Create UMLS Source Vocabulary transitive closures for Poincare embedding.
    
    Author: Louis Qin <louisqin@ucla.edu> or <qyl0509@icloud.com>
"""

import os
import argparse
import pandas as pd

def create_transitive_closure(mrhier_path, closure_dir, source_vocab):
    """Create a transitive closure of (hyponym, hypernym)"""
    mrhier = pd.read_csv(mrhier_path, sep='|', dtype=object)

    source = mrhier[mrhier['SAB'] == source_vocab]

    edges = set()
    for index, row in source.iterrows():
        if pd.isna(row['PTR']):
            continue
        parents = row['PTR'].split('.')
        for i, p in enumerate(parents):
            for c in parents[1:]:
                edges.add((c, p))
            edges.add((row['AUI'], p))

    closure = pd.DataFrame(list(edges), columns=['id1', 'id2'])
    closure['weight'] = 1
    closure.to_csv(os.path.join(closure_dir, f"{source_vocab}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transitive closure')
    parser.add_argument('-v', '--source_vocab', required=True, help='UMLS source vocabulary SAB')
    opt = parser.parse_args()

    data_dir = os.environ['DATA_DIR']
    mrhier_path = os.path.join(data_dir, 'hyperbolic', 'MRHIER_filled.csv')
    closure_dir = os.path.join(data_dir, 'hyperbolic', 'transitive_closure')

    create_transitive_closure(mrhier_path, closure_dir, opt.source_vocab)