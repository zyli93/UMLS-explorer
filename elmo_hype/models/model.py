from typing import Dict, List, Tuple, Union, Optional

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.language_model import LanguageModel, _SoftmaxLoss
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Perplexity

TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]

@Model.register("hyperbolic_tuned_language_model")
class HyperbolicTunedLanguageModel(LanguageModel):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        contextualizer: Seq2SeqEncoder,
        hyperbolic_embedder: TextFieldEmbedder,
        hyperbolic_encoder: Seq2VecEncoder,
        hyperbolic_weight: float,
        dropout: float = None,
        num_samples: int = None,
        sparse_embeddings: bool = False,
        bidirectional: bool = False,
        initializer: InitializerApplicator = None,
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(
            vocab,
            text_field_embedder,
            contextualizer,
            dropout,
            num_samples,
            sparse_embeddings,
            bidirectional,
            initializer,
            regularizer
        )

        # reinitialize self._softmax_loss to change default namespace 'token'
        if num_samples is not None:
            self._softmax_loss = SampledSoftmaxLoss(
                num_words=vocab.get_vocab_size(namespace='euclidean'),
                embedding_dim=self._forward_dim,
                num_samples=num_samples,
                sparse=sparse_embeddings,
            )
        else:
            self._softmax_loss = _SoftmaxLoss(
                num_words=vocab.get_vocab_size(namespace='euclidean'), embedding_dim=self._forward_dim
            )

        # initialize hyperbolic components
        self._hyperbolic_embedder = hyperbolic_embedder
        self._hyperbolic_encoder = hyperbolic_encoder
        self._hyperbolic_encoding_loss = torch.nn.MSELoss()
        self._hyperbolic_weight = hyperbolic_weight

    def forward(
        self, 
        source: TextFieldTensors,
        hyperbolic_tokens: TextFieldTensors = None,
        hyperbolic_phrase: TextFieldTensors = None
    ) -> Dict[str, torch.Tensor]:
        # forward pass on language model
        return_dict = super().forward(source)
        
        if hyperbolic_phrase is not None:
            # forward pass on hyperbolic encoder
            euclidean_embeddings = self._text_field_embedder(hyperbolic_tokens)
            hyperbolic_embedding = self._hyperbolic_embedder(hyperbolic_phrase)
            hyperbolic_embedding = hyperbolic_embedding.squeeze()

            hyperbolic_pred = self._hyperbolic_encoder(
                                    euclidean_embeddings, 
                                    get_text_field_mask(hyperbolic_tokens)
                                )
            hyperbolic_encoding_loss = self._hyperbolic_encoding_loss(hyperbolic_pred, hyperbolic_embedding)

            # aggregate loss
            return_dict['loss'] += hyperbolic_encoding_loss * self._hyperbolic_weight

        return return_dict