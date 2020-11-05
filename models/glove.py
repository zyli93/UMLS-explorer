"""
    Adapted from https://nlpython.com/implementing-glove-model-with-pytorch/
"""

import os
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PMC_DIR = "../../../pmc_clean"
MEDLINE_DIR = "../../../medline_clean"
EMBED_DIM = 128
N_EPOCHS = 100
BATCH_SIZE = 2048 * 8
X_MAX = 100
ALPHA = 0.75

class GloveDataset(Dataset):
    """ GloVe dataset

    Attributes:
        embeds: pretrained poincare embedding
        id2phrase: defaultdict mapping id to phrase
        id2words: defaultdict mapping id to cleaned words
        phrase2id: defaultdict mapping phrase to id
        vocab: defaultdict mapping word to a list of phrase ids
    """
    def __init__(self, window_size=5):
        self._window_size = window_size
        self._tokens = list()
        self._word_counter = Counter()

    def load(self, text):
        tokens = text.split()
        self._tokens.extend(tokens)
        self._word_counter.update(tokens)
        
    def create(self):
        print("STATUS: Tokenizing words to ids...")
        self._word2id = {w:i+1 for i, (w,_) in enumerate(self._word_counter.most_common())}
        self._word2id.update({'<PAD>':0})
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        self._id_tokens = [self._word2id[w] for w in self._tokens]

        self._create_coocurrence_matrix()
        
    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)
        print("STATUS: Creating coocurrence matrix...")
        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j-i)
                    
        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()
        
        print("STATUS: \tReducing cooccurrence matrix memory usage...")
        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)
                
        self._i_idx = torch.LongTensor(self._i_idx)
        self._j_idx = torch.LongTensor(self._j_idx)
        self._xij = torch.FloatTensor(self._xij)
        self.len = self._i_idx.shape[0]
    
    def __getitem__(self, index):
        return self._xij[index], self._i_idx[index], self._j_idx[index]

    def __len__(self):
        return self.len

class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.wj = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.bi = nn.Embedding(num_embeddings, 1, padding_idx=0)
        self.bj = nn.Embedding(num_embeddings, 1, padding_idx=0)
        
        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.wi.weight.data[0].zero_()
        self.wj.weight.data[0].zero_()
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()
        
    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        return x

def weight_func(x, x_max=X_MAX, alpha=ALPHA):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx.cuda()  

def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss).cuda()

def train(loader, model, optimizer, epochs, device):
    for e in range(1, epochs+1):
        t = tqdm(enumerate(loader), total=len(loader))
        t.set_description("Epoch: {}/{}".format(e, epochs))
        for batch_i, (x_ij, i_idx, j_idx) in t:
            x_ij, i_idx, j_idx = x_ij.to(device), i_idx.to(device), j_idx.to(device)
            optimizer.zero_grad()
            
            outputs = model(i_idx, j_idx)
            weights_x = weight_func(x_ij, X_MAX, ALPHA)
            loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
            
            loss.backward()
            optimizer.step()
            
            t.set_postfix(loss=loss.item())
            
        torch.save(model.state_dict(), "glove.pth")

def load_corpus(dir):
    """Load cleaned corpus into a GloveDataset object
    Args:
        dir: a list of directories including corpus files
    Returns:
        a GloveDataset object
    """
    print("=================================Loading Corpus==================================")
    # create dataset
    dataset = GloveDataset()
    # load dataset
    for d in dir:
        pbar = tqdm(os.listdir(d))
        for filename in pbar:
            if filename.endswith(".txt"):
                pbar.set_description("Loading {}".format(filename))
                with open("{}/{}".format(d, filename)) as f:
                    text = f.read()
                    dataset.load(text)
    # create mappings and cooccurence matrix
    dataset.create()
    print("Summary:")
    print("\tCorpus size: {}".format(len(dataset._tokens)))
    print("\tVocabulary size: {}".format(dataset._vocab_len))
    print("==============================Finish Loading Corpus=============================")
    return dataset

if __name__ == "__main__":
    # create dataset from corpus
    dataset = load_corpus([PMC_DIR])
    # create data loader
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    # create glove model
    glove = GloveModel(dataset._vocab_len, EMBED_DIM)
    
    # move model to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parallel
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     glove = nn.DataParallel(glove)
    glove.to(device)
    
    optimizer = optim.Adagrad(glove.parameters(), lr=0.05)
    # train
    train(train_loader, glove, optimizer, 3, device)