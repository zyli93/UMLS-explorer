from pathlib import Path
import random
import glob
import os
import shutil
import argparse

random.seed(42)

def reservoir_sampling(corpus_dir, n_samples):
    """Sample n_samples amount of files"""
    print("sampling corpus")
    corpus_file_pattern = os.path.join(corpus_dir, '**/*.txt')

    rc = []
    for k, path in enumerate(glob.iglob(corpus_file_pattern, recursive=True)):
        if k < n_samples:
            rc.append(str(path))
        else:
            i = random.randint(0, k)
            if i < n_samples:
                rc[i] = str(path)
    return rc

def create_sample_corpus(sample_paths, dest_corpus_dir):
    """Copy a subset of the corpus and create a dataset"""
    print("creating corpus sample")
    for path in sample_paths:
        shutil.copy2(path, dest_corpus_dir)

def create_corpus_vocab(corpus_dir):
    print("creating corpus vocab")
    corpus_file_pattern = os.path.join(corpus_dir, '**/*.txt')

    vocab = set()
    for path in glob.iglob(corpus_file_pattern, recursive=True):
        with open(path) as fp:
            text = fp.read()
            vocab.update(set(text.split()))
    return vocab

def copy_hyperbolic(data_dir, dest_data_dir, source_vocab):
    print("copying hyperbolic embedding and vocabulary")
    vocab_dir = os.path.join(data_dir, 'hyperbolic', 'vocabs', source_vocab)
    embed_path = os.path.join(data_dir, 'hyperbolic', 'embeddings', f'{source_vocab}_embedding.h5')
    
    dest_vocab_dir = os.path.join(dest_data_dir, 'hyperbolic', 'vocabs', source_vocab)
    dest_embed_dir = os.path.join(dest_data_dir, 'hyperbolic', 'embeddings')
    os.makedirs(dest_vocab_dir, exist_ok=True)
    os.makedirs(dest_embed_dir, exist_ok=True)
    
    euclidean_path = os.path.join(vocab_dir, 'euclidean.txt')
    hyperbolic_path = os.path.join(vocab_dir, 'hyperbolic.txt')

    shutil.copy2(euclidean_path, dest_vocab_dir)
    shutil.copy2(hyperbolic_path, dest_vocab_dir)
    shutil.copy2(embed_path, dest_embed_dir)

def _load_hyperbolic_word_vocab(vocab_dir):
    euclidean_path = os.path.join(vocab_dir, 'euclidean.txt')

    with open(euclidean_path) as fp:
        text = fp.read()
        euclidean_vocab = set(text.split('\n'))

    return euclidean_vocab

def create_allennlp_vocab(data_dir, output_dir, source_vocab):
    """Create an allennlp vocabulary with namespace euclidean and hyperbolic"""
    corpus_dir = os.path.join(data_dir, 'corpus')
    corpus_vocab = create_corpus_vocab(corpus_dir)

    hyperbolic_vocab_dir = os.path.join(data_dir, 'hyperbolic', 'vocabs', source_vocab)
    hyperbolic_word_vocab = _load_hyperbolic_word_vocab(hyperbolic_vocab_dir)
    
    print(f"corpus vocabulary size: {len(corpus_vocab)}")
    print(f"hyperbolic word vocabulary size: {len(hyperbolic_word_vocab)}")
    print(f"size of intersection: {len(corpus_vocab.intersection(hyperbolic_word_vocab))}")

    corpus_vocab.update(hyperbolic_word_vocab)

    with open(os.path.join(output_dir, 'euclidean.txt'), 'w') as fp:
        fp.write('\n'.join(corpus_vocab))

    hyperbolic_phrase_orig_path = os.path.join(hyperbolic_vocab_dir, 'hyperbolic.txt')
    hyperbolic_phrase_dest_path = os.path.join(output_dir, 'hyperbolic.txt')
    shutil.copy2(hyperbolic_phrase_orig_path, hyperbolic_phrase_dest_path)
    
    with open(os.path.join(output_dir, 'non_padded_namespaces.txt'), 'w') as fp:
        fp.write('hyperbolic')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample corpus')
    parser.add_argument('-n', '--n_samples', required=True, type=int, help='number of samples')
    parser.add_argument('-s', '--source-vocab', required=True, help='UMLS source vocabulary')
    opt = parser.parse_args()

    data_dir = os.environ['DATA_DIR']
    corpus_dir = os.path.join(data_dir, 'corpus')
    experiment_dir = os.environ['EXPERIMENT_DIR']
    experiment_data_dir = os.path.join(experiment_dir, 'data')
    experiment_corpus_dir = os.path.join(experiment_data_dir, 'corpus')
    os.makedirs(experiment_corpus_dir, exist_ok=True)

    sample_paths = reservoir_sampling(corpus_dir, opt.n_samples)
    create_sample_corpus(sample_paths, experiment_corpus_dir)
    corpus_vocab = create_corpus_vocab(experiment_corpus_dir)
    
    copy_hyperbolic(data_dir, experiment_data_dir, opt.source_vocab)

    output_dir = os.path.join(experiment_data_dir, 'vocab')
    os.makedirs(output_dir, exist_ok=True)

    create_allennlp_vocab(experiment_data_dir, output_dir, opt.source_vocab)