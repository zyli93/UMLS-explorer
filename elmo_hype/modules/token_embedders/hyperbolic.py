import json
from typing import Dict, Tuple, TYPE_CHECKING

import torch
from allennlp.common import Params

from allennlp.common.checks import ConfigurationError
from allennlp.data import TokenIndexer, Token
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import EmptyEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

# Importing at runtime results in a circular dependency.
if TYPE_CHECKING:
    from allennlp_models.lm.models.language_model import LanguageModel


@TokenEmbedder.register("hyperbolic_token_embedder")
class HyperbolicTokenEmbedder(TokenEmbedder):
    def __init__(
        self,
        archive_file: str,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        # Import here to avoid circular dependency.
        from allennlp.models.archival import load_archive

        # Load LM and the associated config.
        archive = load_archive(archive_file)
        self._lm: LanguageModel = archive.model
        self._lm.delete_softmax()
        config = archive.config
        dict_config = config.as_dict(quiet=True)
        
        # Extract the name of the tokens that the LM was trained on.
        text_field_embedder = dict_config["model"]["text_field_embedder"]
        text_field_embedder = TextFieldEmbedder.from_params(Params(text_field_embedder))
        if not isinstance(text_field_embedder, BasicTextFieldEmbedder):
            raise ConfigurationError(
                f"Language model from {archive_file} uses a non-standard TextFieldEmbedder!"
            )
        non_empty_embedders = [
            name
            for name, token_embedder in text_field_embedder._token_embedders.items()
            if not isinstance(token_embedder, EmptyEmbedder)
        ]

        if len(non_empty_embedders) == 0:
            # Only empty embedders were contained in the language model
            # We need at least one non-empty embedder in the language model
            raise ConfigurationError(
                f"Language model from {archive_file} trained with only empty embedders!"
            )
        elif len(non_empty_embedders) > 1:
            raise ConfigurationError(
                f"Language model from {archive_file} trained with multiple non-empty embedders!"
            )

        self._token_name = non_empty_embedders[0]

        for param in self._lm.parameters():
            param.requires_grad = requires_grad

    def forward(
        self,  # type: ignore
        corpus_tokens: torch.Tensor,
        hyperbolic_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            corpus_tokens: dummy tokens
        """     

        source = {self._token_name: {"token_characters": corpus_tokens}}
        hyperbolic_tokens = {self._token_name: {"token_characters": hyperbolic_tokens}}
        result_dict = self._lm(source, hyperbolic_tokens)

        return result_dict["hyperbolic_encoding"]
        