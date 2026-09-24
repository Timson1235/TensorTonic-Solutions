import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    half_dim = d_model // 2

    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(half_dim)
    denominator = base ** (2 * i / d_model)
    
    devided = pos / denominator
    
    matrix = np.zeros((seq_len, d_model))
    
    # 1. Alle geraden Paare mit sin und cos füllen (mit exakter Grenze)
    matrix[:, 0 : 2 * half_dim : 2] = np.sin(devided)
    matrix[:, 1 : 2 * half_dim : 2] = np.cos(devided)
    
    # 2. Falls d_model ungerade ist: Die allerletzte Spalte kriegt noch den sin
    if d_model % 2 != 0:
        i_last = d_model // 2
        denom_last = base ** (2 * i_last / d_model)
        matrix[:, -1] = np.sin(pos.squeeze() / denom_last)
        
    return matrix

