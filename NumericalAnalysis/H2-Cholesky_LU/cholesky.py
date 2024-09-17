import numpy as np
import scipy
import scipy.linalg
import numba

@numba.jit(nopython=True)
def cholesky(A):
    """
    Fonction factorisant une matrice symétrique définie positive A sous la forme LL^T
    La décomposition est faite inplace pour L afin d'économiser de la mémoire
    @param: A: matrice n*n symétrique définie positive

    @return: L: matrice triangulaire inférieure
    """

    n = len(A)
    for k in range(n):
        A[k, k] = np.sqrt(A[k, k])
        for i in range(k+1, n):
            A[i, k] = A[i, k] / A[k, k]

        for j in range(k+1, n):
            for i in range(j, n):
                A[i, j] -= A[i, k]*A[j, k]

    for i in range(n):
        for j in range(i+1, n):
            A[i, j] = 0

    return A






