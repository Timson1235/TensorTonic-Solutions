import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.maximum(np.asarray(x), 0)
    return np.asarray(x)