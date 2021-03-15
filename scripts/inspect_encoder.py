"""
    Helper script to analyze embedding space
"""
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.linalg as LA
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers.token_class import Token
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from elmo_hype.dataset_readers.dataset_reader import UMLSDatasetReader
from elmo_hype.models.model import HyperbolicTunedLanguageModel
from elmo_hype.models.encoder import Euc2HypeEncoder
from elmo_hype.modules.token_embedders.hyperbolic import HyperbolicTokenEmbedder

lm_model_file = "experiments/test/elmo_htlm_reparam_hypeL1_0.5/model.tar.gz"

class Inspector:
    def __init__(self, lm_model_file):
        self.lm_embedder = HyperbolicTokenEmbedder(archive_file=lm_model_file)
        if torch.cuda.device_count():
            self.lm_embedder = self.lm_embedder.cuda()
        # namespace doesn't seem to matter
        self.indexer = ELMoTokenCharactersIndexer(namespace='whatever') 
        self.vocab = self.lm_embedder._lm.vocab
        self.target_embeddings = self.lm_embedder._lm._hyperbolic_embedder._token_embedders['tokens'].weight
        self.dummy_input = self._create_dummy_input()

    def _create_dummy_input(self):
        sentence = "<S> placeholder </S>"
        tokens = [Token(word) for word in sentence.split()]
        character_indices = self.indexer.tokens_to_indices(
            tokens, 
            self.vocab
            )["elmo_tokens"]
        indices_tensor = torch.LongTensor([character_indices])
        return indices_tensor

    def _encode_concepts(self, concepts):
        concept_tensors = []
        for concept in concepts:
            concept = [Token(word) for word in concept.split()]
            concept_indices = self.indexer.tokens_to_indices(
                concept, 
                self.vocab
                )["elmo_tokens"]
            concept_tensors.append(torch.LongTensor(concept_indices))
        return concept_tensors

    def encode_all_concepts(self, batch_size=512):
        hyperbolic_embeddings = []
        concepts = list(self.vocab.get_token_to_index_vocabulary(namespace="hyperbolic").keys())
        for i in tqdm(range(0, len(concepts), batch_size)):
            concept_tensors = self._encode_concepts(concepts[i: i+batch_size])
            padded_concept_tensors = pad_sequence(concept_tensors, batch_first=True)
            if torch.cuda.device_count():
                self.dummy_input = self.dummy_input.cuda()
                padded_concept_tensors = padded_concept_tensors.cuda()
            hyperbolic_embeddings.append(self.lm_embedder(self.dummy_input, padded_concept_tensors))
        hyperbolic_embeddings = torch.cat(hyperbolic_embeddings)
        return hyperbolic_embeddings

    @staticmethod
    def compute_pca(self, embeddings):
        u, s, v = torch.pca_lowrank(embeddings)
        return torch.matmul(embeddings, v[:, :2])

    @staticmethod
    def plot_embeddings(embeddings, img_name):
        embeddings = Inspector.compute_pca(embeddings)
        embeddings = embeddings.cpu().numpy()

        fig, ax = plt.subplots()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
        ax.add_patch(circ)
        plt.scatter(*zip(*embeddings), alpha=0.5)
        plt.savefig(img_name)
    
    @staticmethod
    def print_norm_statistics(embeddings):
        norms = LA.norm(embeddings, dim=1).cpu().numpy()
        print(np.mean(norms), np.std(norms))
