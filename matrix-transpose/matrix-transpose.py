import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    m = len(A)
    n = len(A[0])
    AT = np.zeros([n,m])

    for i in range (m):
        for j in range (n):
            AT[j][i] = A[i][j]

    return AT
    
