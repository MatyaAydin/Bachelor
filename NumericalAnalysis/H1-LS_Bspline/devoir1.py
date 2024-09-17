import numpy as np
import numba


@numba.jit(nopython=True)
def qr(A):
    """
    Applies modified Gram-Schmidt algorithm given during lecture 2
    factorizes A as QR. Q is a unitary matrix and R is upper triangular
    @param:A: m*n matrix

    @return:-Q: m*n unitary matrix
            -R: n*n upper triangular matrix
    """

    m,n =  np.shape(A)
    Q = A.copy()
    R = np.zeros((n, n))
    
    for k in range(n):
        R[k,k] = norm_numba(Q[:,k])
        if R[k, k] != 0:
            Q[:,k] /= R[k,k]
        for j in range(k+1, n):
            R[k,j] = dot_prod(Q[:,k], Q[:,j])
            Q[:,j] -= R[k,j] * Q[:,k]

    return Q,R


def lstsq(A, B):
    """
    Solve the least squares optimization problem min ||AX-B|| by using the QR factorization
    @param:-A: m*n matrix
    @param:-B: m*p matrix

    @return X: n*p matrix which satisfies the normal equations
    """
    Q,R = qr(A)
    m = len(A)
    n = len(R)
    p = len(B[0])
    X = np.zeros((n, p))
    for i in range(p):
        b = B[:,i]
        b = Q.T @ b
        X[:,i] = diagonal_system_solver(R, b)


    return X


def diagonal_system_solver(A, b):
    """
    Applies a backsubstitution to efficiently solve the diagonal system Ax = b
    @param: A: n*n upper triangular matrix
    @param: b: column vector of coefficient

    @return: x: vector of solutions
    """
    n = len(b)
    x = np.zeros((n))

    
    for i in range(n): #n-1, -1, -1
        k = n - 1 - i
        #print(k)
        x[k] = b[k]
        for j in range(k+1, n):
            x[k] -= x[j]*A[k, j]
        if A[k, k] != 0:
            x[k] = x[k]/A[k, k]


    return x


@numba.jit(nopython=True, parallel=True)
def norm_numba(x):
    """
    Computes the l_2 norm of a vector
    @param: x: 1D numpy array
    @return norm: l_2(x)
    """
    norm = 0
    n = len(x)
    for i in numba.prange(n):
        norm += x[i]**2

    return np.sqrt(norm)


@numba.jit(nopython=True, parallel=True)
def dot_prod(v1, v2):
    tot = 0
    for i in numba.prange(len(v1)):
        tot += v1[i]*v2[i]
    return tot