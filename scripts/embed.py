"""
    Example usage of pre-trained ELMo.
"""

from allennlp_models.lm.modules.token_embedders.bidirectional_lm import BidirectionalLanguageModelTokenEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers.token_class import Token
import torch

from elmo_hype.dataset_readers.dataset_reader import UMLSDatasetReader
from elmo_hype.models.model import HyperbolicTunedLanguageModel
from elmo_hype.models.encoder import Euc2HypeEncoder

lm_model_file = "experiments/test/elmo_htlm_reparam_hypeL1_0.5/model.tar.gz"

sentence = "<S> It is raining . </S>"
tokens = [Token(word) for word in sentence.split()]

lm_embedder = BidirectionalLanguageModelTokenEmbedder(
    archive_file=lm_model_file,
    # bos_eos_tokens=None,
    # remove_bos_eos=False
)

indexer = ELMoTokenCharactersIndexer(namespace='bogus') # namespace doesn't seem to matter
vocab = lm_embedder._lm.vocab
character_indices = indexer.tokens_to_indices(tokens, vocab)["elmo_tokens"]

# Batch of size 1
indices_tensor = torch.LongTensor([character_indices])

# Embed and extract the single element from the batch.
embeddings = lm_embedder(indices_tensor)[0]
print(embeddings)