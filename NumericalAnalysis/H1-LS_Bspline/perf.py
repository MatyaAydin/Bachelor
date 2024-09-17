from time import perf_counter as clock
from devoir1 import *
import numpy as np
import numba
import matplotlib.pyplot as plt



def norm(x):
    """
    Computes the l_2 norm of a vector
    @param: x: 1D numpy array
    @return norm: l_2(x)
    """
    norm = 0
    n = len(x)
    for i in range(n):
        norm += x[i]**2

    return np.sqrt(norm)

def qrNoNumba(A):
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
        R[k,k] = norm(Q[:,k])
        if R[k, k] != 0:
            Q[:,k] /= R[k,k]
        for j in range(k+1, n):
            R[k,j] = Q[:,k] @ Q[:,j]
            Q[:,j] -= R[k,j] * Q[:,k]

    return Q,R
def qr_householder(A):
    m, n = A.shape
    Q = np.eye(m)
    R = np.copy(A)
    for i in range(0, n-1):
        x = norm(R[i:, i])
        e = np.zeros_like(R[i:, [i]])
        e[0] = x
        u = R[i:, [i]] - e 
        v = u/norm(u)
        Qn = np.eye(m-i) - (2*np.dot(v, v.T))
        Qn = np.block([[np.eye(i), np.zeros((i, m-i))],
                        [np.zeros((m-i,i)), Qn]])
        R = np.dot(Qn, R)
        Q = np.dot(Q, Qn.T)
    return Q, R



 #cas m = n
m = 1000
same_dim_numba = np.zeros((m), dtype='float64')
same_dim_no_numba = np.zeros((m), dtype='float64')
same_dim_householder = np.zeros((m), dtype='float64')
diff_dim = np.zeros((m), dtype='float64')


A = np.random.rand(10, 10)
Q, R = qr(A)
#Q, R = qr_givens(A)

for i in range(1, m+1):
    A = np.random.rand(i, i)
    t = clock()
    Q, R = qr(A)
    dt = clock()-t
    same_dim_numba[i-1] = dt

    t = clock()
    Q, R = qrNoNumba(A)
    dt = clock()-t
    same_dim_no_numba[i-1] = dt

    t = clock()
    Q, R = qr_householder(A)
    dt = clock()-t
    same_dim_householder[i-1] = dt







Csquare = 1e-4
abs = list(range(1, m+1))
big_Osquare = [Csquare*i**2 for i in abs]

C_cube = 1e-7
big_Ocube = [C_cube*i**3 for i in abs]



plt.plot(abs, same_dim_numba, label="$qr$ Gram-Schmidt, avec numba")
plt.plot(abs, same_dim_no_numba, label="$qr$ Gram-Schmidt, sans numba")
plt.plot(abs, same_dim_householder, label="$qr$ Householder sans numba")

plt.plot(abs, big_Ocube, label="$\mathcal{O}(n^3)$", c="black", linestyle=":")
#plt.plot(abs, big_Osquare, label="$\mathcal{O}(n^2)$", c="red", linestyle=":")

plt.xscale("log")
plt.yscale("log")
plt.grid(True, axis='both', linestyle='--')
plt.title("Temps d'exécution de la fonction $qr$ pour $m = n$")
plt.legend(loc='upper left')
plt.xlabel("Dimensions de A $[-]$")
plt.ylabel("Temps d'exécution $[s]$")
#plt.savefig("perf_same_dim_householder.pdf")
plt.show()





#cas m constant: quadratique wrt n

m = 600
n = 600
same_dim_numba = np.zeros((n), dtype='float64')
same_dim_no_numba = np.zeros((n), dtype='float64')
same_dim_householder = np.zeros((n), dtype='float64')
diff_dim = np.zeros((n), dtype='float64')


A = np.random.rand(10, 10)
Q, R = qr(A)

for i in range(1, n+1):
    A = np.random.rand(m, i)
    t = clock()
    Q, R = qr(A)
    dt = clock()-t
    same_dim_numba[i-1] = dt

    t = clock()
    Q, R = qrNoNumba(A)
    dt = clock()-t
    same_dim_no_numba[i-1] = dt

    t = clock()
    Q, R = qr_householder(A)
    dt = clock()-t
    same_dim_householder[i-1] = dt


Csquare = 1e-4
abs = list(range(1, m+1))
big_Osquare = [Csquare*i**2 for i in abs]




plt.plot(abs, same_dim_numba, label="$qr$ Gram-Schmidt, avec numba")
plt.plot(abs, same_dim_no_numba, label="$qr$ Gram-Schmidt, sans numba")
plt.plot(abs, same_dim_householder, label="$qr$ Householder sans numba")

plt.plot(abs, big_Osquare, label="$\mathcal{O}(n^2)$", c="black", linestyle=":")

plt.xscale("log")
plt.yscale("log")
plt.grid(True, axis='both', linestyle='--')
plt.title("Temps d'exécution de la fonction $qr$ pour $m = {}$".format(m))
plt.legend(loc='upper left')
plt.xlabel("n $[-]$")
plt.ylabel("Temps d'exécution $[s]$")
#plt.savefig("perf_quadratique_m={}.pdf".format(m))
plt.show()




#cas n constant: linéaire wrt m

m = 500
n= 400
same_dim_numba = np.zeros((m - n +1), dtype='float64')
same_dim_no_numba = np.zeros((m - n + 1), dtype='float64')
same_dim_householder = np.zeros((m-n +1), dtype='float64')
diff_dim = np.zeros(( m - n + 1), dtype='float64')


A = np.random.rand(10, 10)
Q, R = qr(A)

for i in range(n, m+1):
    A = np.random.rand(i, n)
    t = clock()
    Q, R = qr(A)
    dt = clock()-t
    same_dim_numba[i - n] = dt

    t = clock()
    Q, R = qrNoNumba(A)
    dt = clock()-t
    same_dim_no_numba[i - n] = dt

    t = clock()
    Q, R = qr_householder(A)
    dt = clock()-t
    same_dim_householder[i-1 - n] = dt



Clin = 1e-2
abs = list(range(n, m+1))
big_Olin = [Clin*i for i in abs]




plt.plot(abs, same_dim_numba, label="$qr$ Gram-Schmidt, avec numba")
plt.plot(abs, same_dim_no_numba, label="$qr$ Gram-Schmidt, sans numba")
plt.plot(abs, same_dim_householder, label="$qr$ Householder sans numba")

plt.plot(abs, big_Olin, label="$\mathcal{O}(m)$", c="black", linestyle=":")

plt.xscale("log")
plt.yscale("log")
plt.grid(True, axis='both', linestyle='--')
plt.title("Temps d'exécution de la fonction $qr$ pour $n = {}$".format(n))
plt.legend(loc='upper left')
plt.xlabel("m $[-]$")
plt.ylabel("Temps d'exécution $[s]$")
#plt.savefig("perf_lin.pdf")
plt.show()











