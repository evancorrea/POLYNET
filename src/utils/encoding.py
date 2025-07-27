import numpy as np

#define one_hot encoding function
def one_hot(seq):
    """Convert RNA string to (4x201) one-hot encoding, float32 array."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nt in enumerate(seq.upper()):
        idx = mapping.get(nt, 4)
        if idx < 4:
            arr[idx,i] = 1.0
    return arr