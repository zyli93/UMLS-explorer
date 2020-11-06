from pathlib import Path
import random
import glob
import os
import shutil
import argparse

random.seed(42)

def reservoir_sampling(corpus_dir, n_samples):
    """Sample n_samples amount of files"""
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

def create_sample_corpus(sample_paths, output_dir):
    """Copy a subset of the corpus and create a dataset"""
    for path in sample_paths:
        shutil.copy2(path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample corpus')
    parser.add_argument('-n', '--n_samples', required=True, type=int, help='number of samples')
    opt = parser.parse_args()

    data_dir = os.environ['DATA_DIR']
    corpus_dir = os.path.join(data_dir, 'corpus')

    experiment_dir = os.environ['EXPERIMENT_DIR']
    experiement_corpus_dir = os.path.join(experiment_dir, 'data', 'corpus')
    
    try:
        os.makedirs(experiement_corpus_dir)
    except:
        pass

    sample_paths = reservoir_sampling(corpus_dir, opt.n_samples)
    create_sample_corpus(sample_paths, experiement_corpus_dir)