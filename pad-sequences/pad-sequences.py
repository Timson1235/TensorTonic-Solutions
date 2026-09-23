import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    N = len(seqs)
    L = max_len if max_len is not None else (max((len(row) for row in seqs), default=0))
    
    if N == 0:
        return np.empty((0, 0), dtype=int)
    L = max_len if max_len is not None else (max((len(row) for row in seqs), default=0) if seqs else 0)
    
    padded_rows = [list(row)[:L] + [pad_value] * max(0, L - len(list(row)[:L])) for row in seqs]
    return np.array(padded_rows)