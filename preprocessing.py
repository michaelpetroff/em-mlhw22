from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import xml.etree.ElementTree as ET
from collections import Counter


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    sentence_pairs, alignments = [], []

    text = open(filename).read()
    tree = ET.fromstring(text.replace('&', '&amp;'))

    for e in tree.iter('s'):
        source = e.find('english').text.split()
        target = e.find('czech').text.split()
        sentence_pairs.append(SentencePair(source, target))

        sure_text = e.find('sure').text
        sure = [tuple(map(int, pair.split('-'))) for pair in sure_text.split()] if sure_text else []
        possible_text = e.find('possible').text
        possible = [tuple(map(int, pair.split('-'))) for pair in possible_text.split()] if possible_text else []
        alignments.append(LabeledAlignment(sure, possible))
    
    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_counter, target_counter = Counter(), Counter()
    for pair in sentence_pairs:
        source_counter.update(pair.source)
        target_counter.update(pair.target)
    
    if freq_cutoff is not None:
        source_counter = Counter(dict(source_counter.most_common(freq_cutoff)))
        target_counter = Counter(dict(target_counter.most_common(freq_cutoff)))
    
    source_dict = dict(zip(list(source_counter), range(len(source_counter))))
    target_dict = dict(zip(list(target_counter), range(len(target_counter))))
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    for pair in sentence_pairs:
        try:
            source_tokens = np.array(list(map(lambda x: source_dict[x], pair.source)))
            target_tokens = np.array(list(map(lambda x: target_dict[x], pair.target)))
        except KeyError:
            continue
        tokenized_sentence_pairs.append(TokenizedSentencePair(source_tokens, target_tokens))
    return tokenized_sentence_pairs
