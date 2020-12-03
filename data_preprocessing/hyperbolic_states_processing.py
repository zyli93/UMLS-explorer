import os
import argparse
import h5py
import torch
import numpy as np
import pandas as pd

def load_mrconso(mrconso_path):
    """load MRCONSO to map AUIs to STRs"""
    print(f"loading {mrconso_path}")
    mrconso = pd.read_csv(mrconso_path, sep='|', header=None, dtype=object)
    mrconso = mrconso.drop(18, axis=1) # last column is meaningless because the entry ends with '|'
    mrconso.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"]
    return mrconso

def load_checkpoint(checkpoint_path):
    """load poincare checkpoint"""
    print("loading {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def transform(checkpoint, mrconso):
    """tranform pytorch states checkpoint to vocab and embedding"""
    assert len(checkpoint['objects']) == len(checkpoint['embeddings']), "mismatch between number of AUIs and embeddings"

    print("transforming checkpoint")
    phrase2embedding = dict()
    phrase2aui = dict()
    duplicate_phrases = set()

    for aui, embedding in zip(checkpoint['objects'], checkpoint['embeddings']):
        matches = mrconso[mrconso['AUI'] == aui]
        n = matches.shape[0]
        if n >= 1:
            phrase = matches['STR'].iloc[0]
            phrase = phrase.lower()
            if phrase in phrase2embedding:
                duplicate_phrases.add(phrase)
                print(f"Duplicate STR {phrase} AUI {aui}")
            else:
                phrase2embedding[phrase] = embedding
                phrase2aui[phrase] = aui
                if n > 1:
                    print(f"AUI {aui} has more than one STR, defaulting to the first one")
                    print(matches)
        else:
            print("AUI missing in MRCONSO", aui)

    auis = []
    phrases =['EUCLID-HYPER-ENCODER-PLACEHOLDER']
    embeddings = [np.zeros(len(list(phrase2embedding.values())[0]))]

    for phrase in phrase2embedding:
        if phrase not in duplicate_phrases:
            phrases.append(phrase)
            auis.append(phrase2aui[phrase])
            embeddings.append(phrase2embedding[phrase])

    words = {w for p in phrases for w in p.split()} # TODO: ideally this should use allennlp
    embeddings = np.stack(embeddings)

    print(f"number of embeddings preserved: {len(embeddings)} out of {len(checkpoint['embeddings'])}")

    return {
        'auis': auis,
        'words': words,
        'phrases': phrases,
        'embedding': embeddings
    }

def save(auis, words, phrases, embedding, data_dir, source_vocab, tsv=False):
    """save embeddings hdf5 and words and phrase vocabularies; optionally create tsv for visualization"""
    print("saving embedding")
    embeddings_dir = os.path.join(data_dir, 'hyperbolic', 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    embedding_h5_path = os.path.join(embeddings_dir, f'{source_vocab}_embedding.h5')

    with h5py.File(embedding_h5_path, 'w') as fin:
        fin.create_dataset('embedding', data=embedding)
    
    print("saving vocabs")
    vocab_dir = os.path.join(data_dir, 'hyperbolic', 'vocabs', source_vocab)
    os.makedirs(vocab_dir, exist_ok=True)
    euclidean_path = os.path.join(vocab_dir, 'euclidean.txt')
    hyperbolic_path = os.path.join(vocab_dir, 'hyperbolic.txt')

    with open(euclidean_path, "w") as fp:
        fp.write("\n".join(words))
    with open(hyperbolic_path, "w") as fp:
        fp.write("\n".join(phrases))

    if tsv:
        print("saving tsv")
        tsv_dir = os.path.join(data_dir, 'hyperbolic', 'tsvs', source_vocab)
        os.makedirs(tsv_dir, exist_ok=True)

        embedding_tsv_path = os.path.join(tsv_dir, f"{source_vocab}_embedding.tsv")
        np.savetxt(embedding_tsv_path, embedding, delimiter='\t')

        labels_tsv_path = os.path.join(tsv_dir, f"{source_vocab}_labels.tsv")
        with open(labels_tsv_path, "w") as fp:
            print('AUI\tString', file=fp)
            fp.write('\n'.join(['\t'.join([a, p]) for a, p in zip(auis, phrases)]))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process hyperbolic torch states')
    parser.add_argument('-s', '--source-vocab', required=True, help='UMLS source vocabulary')
    parser.add_argument('-t', '--export-tsv', default=True, help='export tsvs for visualization')
    opt = parser.parse_args()

    data_dir = os.environ['DATA_DIR']
    mrconso_path = os.environ['MRCONSO_PATH']
    checkpoint_path = os.path.join(data_dir, 'hyperbolic', 'torch_states', opt.source_vocab, f'{opt.source_vocab}.pth.best')

    mrconso = load_mrconso(mrconso_path)
    checkpoint = load_checkpoint(checkpoint_path)

    transformed_data = transform(checkpoint, mrconso)
    save(**transformed_data, data_dir=data_dir, source_vocab=opt.source_vocab, tsv=opt.export_tsv)