"""
Implémentation de l'algorithme de décomposition LU sans pivotage

"""

import numpy as np
import numba

@numba.jit(nopython=True)
def lu(A):
    """
    Fonction factorisant une matrice inversible A sous la forme LU
    La décomposition est faite inplace pour U afin d'économiser de la mémoire
    @param: A: matrice n*n inversible

    @return: L: matrice n*n triangulaire inférieure avec une diagonale unitaire
    @return: U: matrice triangulaire supérieure

    """
    n = len(A)
    L = np.eye(n)
   
    for k in range(n-1):
        pivot = A[k, k]
        for j in range(k+1, n):
            L[j, k] = A[j,k]/pivot

            A[j,k:] -= L[j, k]*A[k, k:]


    return L,A

#Montrer qu'on peut pas betement paralleliser une LU sans traiter les dépendances entre les itérations
if __name__ == "__main__":
    @numba.jit(nopython=True, parallel=True)
    def lu_parallel(A):

        n = len(A)
        L = np.eye(n)
    
        for k in numba.prange(n-1):
            pivot = A[k, k]
            for j in numba.prange(k+1, n):
                L[j, k] = A[j,k]/pivot
                for i in numba.prange(k, n):
                    A[j,i] -= L[j, k]*A[k, i]


        return L,A


    rep = {True: 0, False: 0}
    for i in range(500):
        A = np.random.rand(3, 3)
        sdp = A.T @ A
        #On copie car la LU est inplace
        sdpbis = np.copy(sdp)
        sdp_cmp = np.copy(sdp)

        L, U = lu_parallel(sdpbis)
        A = L @ U
        rep[np.allclose(A, sdp)] +=1

    print(rep)
    