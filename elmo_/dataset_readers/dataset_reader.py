"""
    Credit to SimpleLanguageModelingDatasetReader from AllenNLP
    Reference: https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/data/dataset_readers/simple_language_modeling.py
"""

from typing import Dict, Iterable, Union, Optional, List
import logging
import math
from collections import defaultdict
import glob

from overrides import overrides

from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("umls")
class UMLSDatasetReader(DatasetReader):
    """
    Reads sentences, one per line, for language modeling. This does not handle arbitrarily formatted
    text with sentences spanning multiple lines.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sentences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    max_sequence_length: ``int``, optional
        If specified, sentences with more than this number of tokens will be dropped.
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to the ``TextField``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to the ``TextField``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,                # NOTE: namespace corpus
                 hyperbolic_phrase_indexers: Dict[str, TokenIndexer] = None,    # NOTE: namespace hyperbolic
                 max_sequence_length: int = None,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 max_file_count: int = -1) -> None:
        super().__init__(True)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._hyperbolic_phrase_indexers = hyperbolic_phrase_indexers or {"tokens": SingleIdTokenIndexer()}

        if max_sequence_length is not None:
            self._max_sequence_length: Union[float, Optional[int]] = max_sequence_length
        else:
            self._max_sequence_length = math.inf

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]
        self._max_file_count = max_file_count

        logger.info("Creating SimpleLanguageModelingDatasetReader")
        logger.info("max_sequence_length=%s", max_sequence_length)

    @overrides
    def _read(self, 
              file_path: str) -> Iterable[Instance]:
        # pylint: disable=arguments-differ
        hyperbolic_path = file_path + 'hyperbolic'
        corpus_path = file_path + 'corpus'

        self._map_token_to_hyperbolic_phrases(hyperbolic_path + 'labels.tsv')

        logger.info('Loading corpus data from %s', corpus_paths)
        dropped_instances = 0

        for i, filename in enumerate(glob.iglob(corpus_path + '**/*.txt', recursive=True)):
            if i == self._max_file_count:
                break
            with open(filename) as file:
                for sentence in file:
                    tokens = self._tokenizer.tokenize(sentence)
                    for tok in tokens:
                        if tok.text in self._token2phrases:
                            for phrase in self._token2phrases[tok.text]:
                                instance = self.text_to_instance(sentence, phrase)
                                if instance.fields['source'].sequence_length() <= self._max_sequence_length:
                                    yield instance
                                else:
                                    dropped_instances += 1
        
        if not dropped_instances:
            logger.info(f"No instances dropped.")
        else:
            logger.warning(f"Dropped {dropped_instances} instances.")

        logger.info(f"Loaded corpus from {i} files.")
    
    def _map_token_to_hyperbolic_phrases(self, 
                                         phrase_file: str) -> None:
        self._token2phrases = defaultdict(list)

        logger.info('Loading hyperbolic phrases from %s', phrase_file)
        with open(phrase_file) as file:
            for line in file:
                aui, phrase = file.split('\t')
                tokens = self._tokenizer.tokenize(phrase)
                for tok in tokens:
                    self._token2phrases[tok.text].append(phrase)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: str,
                         hyperbolic_phrase: str = None) -> Instance:
        # pylint: disable=arguments-differ
        tokenized = self._tokenizer.tokenize(sentence)
        tokenized_with_ends = []
        tokenized_with_ends.extend(self._start_tokens)
        tokenized_with_ends.extend(tokenized)
        tokenized_with_ends.extend(self._end_tokens)
        return_instance = Instance({
                'source': TextField(tokenized_with_ends, self._token_indexers)
        })

        if hyperbolic_phrase:
            hyperbolic_tokens = self._tokenizer.tokenize(hyperbolic_phrase)
            return_instance.update({
                'hyperbolic_tokens': TextField(hyperbolic_tokens, self._token_indexers),
                'hyperbolic_phrase': TextField(Token(hyperbolic_phrase), self._hyperbolic_phrase_indexers) 
            })

        return return_instance