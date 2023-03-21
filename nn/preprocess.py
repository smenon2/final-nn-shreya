# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike


def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    seqs = np.array(seqs)
    labels = np.array(labels)

    pos_seqs = seqs[labels == True]
    neg_seqs = seqs[labels == False]

    num_pos = len(pos_seqs)
    num_neg = len(neg_seqs)

    if num_pos < num_neg:
        upsample_cors = np.random.choice(num_pos, num_neg, replace = True)
        upsample_seqs = pos_seqs[upsample_cors]
        max_seqs = neg_seqs
        max_cor = False
        min_cor = True

    elif num_neg < num_pos:
        print("Fewer FALSE")
        upsample_cors = np.random.choice(num_neg, num_pos, replace=True)
        print(upsample_cors)
        upsample_seqs = neg_seqs[upsample_cors]
        max_seqs = pos_seqs
        max_cor = True
        min_cor = False
    else:
        upsample_cors = np.where(labels == True)
        upsample_seqs = pos_seqs[upsample_cors]
        max_seqs = neg_seqs
        max_cor = False
        min_cor = True

    sampled_seqs = np.hstack((upsample_seqs, max_seqs))
    sampled_labels = np.array([min_cor] * len(upsample_seqs) + [max_cor] * len(max_seqs))

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    enc = []

    encoding = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1]}

    # Do the encoding for each string
    for s in seq_arr:
        e = []
        for i in s:
            e.append(encoding[i])
        e = np.array(e)
        e = e.flatten()
        enc.append(e)

    # Return an array version of the final list
    return np.array(enc)
