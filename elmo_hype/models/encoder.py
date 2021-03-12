import torch
from torch import nn
from torch import linalg as LA
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

@Seq2VecEncoder.register("euc2hype")
class Euc2HypeEncoder(Seq2VecEncoder):
    """
        Encode a sequence of Euclidean embeddings into a poincare embedding
        using the reparametrization trick in "Embedding Text in Hyperbolic
        Spaces"

        The Poincare embedding is decomposed into norm and direction which are
        encoded separately from the Euclidean embedding sequence.
    """

    def __init__(
        self,
        norm_encoder: Seq2VecEncoder,
        dir_encoder: Seq2VecEncoder
    ):
        super().__init__()
        self.norm_encoder = norm_encoder
        self.dir_encoder = dir_encoder
        self.fc = nn.Linear(norm_encoder._module.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.BoolTensor
    ) -> torch.Tensor:
        # (n_batch, d_hyper)
        direction = self.dir_encoder(inputs, mask)
        # (n_batch, 1)
        dir_norm = LA.norm(direction, dim=1, keepdim=True)
        # (n_batch, d_hyper) unit vectors
        direction = direction / dir_norm

        # (n_batch, d_norm)
        norm = self.norm_encoder(inputs, mask)
        # (n_batch, 1)
        norm = self.fc(norm)
        norm = self.sigmoid(norm)

        # (n_batch, d_hyper)
        embed_hyper = direction * norm
        
        return embed_hyper
        
