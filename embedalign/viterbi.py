import numpy as np


def get_viterbi(x_batch, y_batch, pa_x, py_xa):
    """Returns the Viterbi alignment for (x, y)"""

    batch_size, longest_y = y_batch.shape
    alignments = np.zeros((batch_size, longest_y), dtype="int64")
    probabilities = np.zeros((batch_size, longest_y), dtype="float32")

    for b, y in enumerate(y_batch):
        for j, yj in enumerate(y):
            if yj == 0:  # Padding
                break
            # TODO: use P(a|x) so that we can have non-uniform priors over alignments
            pyj_xza = py_xa[b, :, yj]
            a_j = pyj_xza.argmax()  # greedy decoder (exact in case of IBM1)
            p_j = pyj_xza[a_j]
            alignments[b, j] = a_j
            probabilities[b, j] = p_j

    return alignments, probabilities
