from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    numer, denom = 0, 0
    for ref, pred in zip(reference, predicted):
        possible_set = set(ref.sure + ref.possible)
        numer += sum(1 if t in possible_set else 0 for t in pred)
        denom += len(pred)
    return numer, denom


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    numer, denom = 0, 0
    for ref, pred in zip(reference, predicted):
        predict_set = set(pred)
        numer += sum(1 if t in predict_set else 0 for t in ref.sure)
        denom += len(ref.sure)
    return numer, denom


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    n1, d1 = compute_precision(reference, predicted)
    n2, d2 = compute_recall(reference, predicted)
    return 1 - (n1 + n2) / (d1 + d2)
