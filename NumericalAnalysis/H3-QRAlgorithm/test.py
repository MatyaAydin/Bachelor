import numpy as np
import scipy
from devoir3 import *



n = 5
nb_tests = 10
eps = 1e-12
max_iter = 1000
identity = np.eye(n, dtype=np.complex128)
didentity = {True:0, False:0}
dschur = {True:0, False:0}
dtriangle = {True:0, False:0}


for _ in range(nb_tests):
    A = np.random.rand(n,n).astype(np.complex128) + 1j*np.random.rand(n,n).astype(np.complex128)

    #Si on veut A ssdp:
    #A = A @ A.T
 

    same = np.zeros((n,n))
    sameSchur = np.zeros((n,n))
    sameQ = np.zeros((n,n))
    same_init = np.zeros((n,n))

    #Pour comparer avec la fonction de scipy:
    eig = np.sort(scipy.linalg.eigvals(A))
    Abis = np.copy(A)
    T, Z = scipy.linalg.schur(A, output="complex")


    #U,k = solve_qr(A, False, eps, max_iter)
    U_shifted,k_shifted  = solve_qr(Abis, True, eps, max_iter)
    my_eig = np.sort(np.diag(Abis))
    
    expected = A @ U_shifted
    actual = U_shifted @ Abis


    actual_init = U_shifted @ Abis @ np.conj(U_shifted.T)


    for i in range(n):
        for j in range(n):
            if np.abs(expected[i,j] - actual[i,j]) < eps:
                same[i,j] = 1
            if np.abs(T[i,j] - Abis[i,j]) < eps:
                sameSchur[i,j] = 1
            if np.abs(U_shifted[i,j] - Z[i,j]) < eps:
                sameQ[i,j] = 1
            if np.abs(A[i,j] - actual_init[i,j]) < eps:
                same_init[i,j] = 1


    dschur[np.allclose(expected, actual, atol=1e-10)] +=1
    didentity[np.allclose(U_shifted @ np.conj(U_shifted.T), identity, atol=1e-10)] +=1
    dtriangle[check_convergence(Abis, eps, n)] += 1


    #print(np.round(Abis, 3), "\n")

    #print(U_shifted @ Abis @ np.conj(U_shifted.T))
    #print(expected - actual)

    #Voir quels éléments sont les mêmes:
    
    """
    print("AU = UT:")
    print(same)
    print("Schur form: ")
    print(sameSchur)
    #print("Q = Z:")
    #print(sameQ)
    print("Gets A back: ")
    print(same_init, "\n")
    """
    
    

print("Schur form: ", dschur)
print("U unitary: ", didentity)
print("Is triangular: ", dtriangle)
    
