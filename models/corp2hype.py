import glove
import poincare
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

CORP_EMB_DIM = 128
HYPE_EMB_DIM = 128
HIDDEN_DIM = 256

PMC_DIR = "../../pmc_clean"
MEDLINE_DIR = "../../medline_clean"
POINCARE_DIR = "../../../poincare/icd10/icd10.pth.best"
N_EPOCHS = 100
BATCH_SIZE = 2048


class Corp2Hype(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Corp2Hype, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.device = None
    
    def init_hidden(self, batch_size):
        '''initialize hidden states to (layers, batch size, hidden him)'''
        return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device), 
                torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq, self.hidden)
        lstm_out, length = pad_packed_sequence(lstm_out)
        avg_pool = torch.sum(lstm_out, dim=0) / length.unsqueeze(1).to(self.device)
        return self.linear(avg_pool)

def generate_corp2hype_train_data(corp_data, hype_data, glove_model, words, is_i, device):
    """Generate training data for Corp2Hype model by keeping words in both vocabularies
    Args:
        corp_data: a GloveDataset object
        hype_data: a PoincareDataset object
        glove_model: a GloveModel object
        words: query words
        is_i: words are from Glove embedding wi
        device: device
    Returns:
        src: a list of phrases - corpus word embeddings
        trg: a list of phrases - poincare phrase embeddings
    """
    src = list()
    trg = list()
    for word in words:
        for word_list, embedding in hype_data.lookup(word):
            if all(w in corp_data._word2id for w in word_list):
                if is_i:
                    src.append(glove_model.wi(torch.LongTensor([corp_data._word2id[w] for w in word_list]).to(device)))
                else:
                    src.append(glove_model.wj(torch.LongTensor([corp_data._word2id[w] for w in word_list]).to(device)))
                trg.append(embedding.to(device))
    return src, trg

def train(corp_data, hype_data, glove_model, corp2hype_model, glove_optim, corp2hype_optim, glove_lossfn, corp2hype_lossfn, epochs, device):
    glove_loader = DataLoader(dataset=corp_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    for e in range(1, epochs+1):
        t = tqdm(enumerate(glove_loader), total=len(glove_loader))
        t.set_description("Epoch: {}/{}".format(e, epochs))
        for batch_i, (x_ij, i_idx, j_idx) in t:

            glove_optim.zero_grad()
            corp2hype_optim.zero_grad()

            # Corp2Hype
            i_words = [corp_data._id2word[i.item()] for i in i_idx if corp_data._id2word[i.item()] in hype_data.vocab]
            j_words = [corp_data._id2word[i.item()] for i in j_idx if corp_data._id2word[i.item()] in hype_data.vocab]

            srci, trgi = generate_corp2hype_train_data(corp_data, hype_data, glove_model, i_words, is_i=True, device=device)
            srcj, trgj = generate_corp2hype_train_data(corp_data, hype_data, glove_model, j_words, is_i=False, device=device)
        
            src = pack_sequence(srci+srcj, enforce_sorted=False)
            trg = torch.cat(trgi+trgj)

            corp2hype_model.hidden = corp2hype_model.init_hidden(len(trg))
            corp2hype_out = corp2hype_model(src)
            corp2hype_loss = corp2hype_lossfn(corp2hype_out.double(), trg)

            corp2hype_loss.backward()
            corp2hype_optim.step()

            # GloVe
            x_ij, i_idx, j_idx = x_ij.to(device), i_idx.to(device), j_idx.to(device)
            
            glove_out = glove_model(i_idx, j_idx)
            weights_x = glove.weight_func(x_ij)
            glove_loss = glove_lossfn(weights_x, glove_out, torch.log(x_ij))
            
            glove_loss.backward()
            glove_optim.step()
            
            t.set_postfix(corp2hype_loss=corp2hype_loss.item(), glove_loss=glove_loss.item())
            
        torch.save(glove_model.state_dict(), "glove.pth")
        torch.save(corp2hype_model.state_dict(), "corp2hype.pth")

if __name__ == "__main__":
    # create datasets
    corp_data = glove.load_corpus([PMC_DIR])
    hype_data = poincare.load_pretrained(POINCARE_DIR)

    # create models
    glove_model = glove.GloveModel(corp_data._vocab_len, CORP_EMB_DIM)
    corp2hype_model = Corp2Hype(CORP_EMB_DIM, HIDDEN_DIM, HYPE_EMB_DIM)
    # move models to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    glove_model = glove_model.to(device)
    corp2hype_model = corp2hype_model.to(device)
    corp2hype_model.device = device
    
    # optimizers
    glove_optim = optim.Adagrad(glove_model.parameters(), lr=0.05)
    corp2hype_optim = optim.Adam(corp2hype_model.parameters(), lr=0.05)

    # loss functions
    glove_lossfn = glove.wmse_loss
    corp2hype_lossfn = nn.MSELoss()
    
    train(corp_data, hype_data, glove_model, corp2hype_model, glove_optim, corp2hype_optim, glove_lossfn, corp2hype_lossfn, 10, device)