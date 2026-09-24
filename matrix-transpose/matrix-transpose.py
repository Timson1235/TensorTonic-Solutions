import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    Matrix = np.array(A)
    matrix_transpose = Matrix.T
    # Write code here
    return matrix_transpose
