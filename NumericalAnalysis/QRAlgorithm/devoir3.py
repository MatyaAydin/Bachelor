import numba
import numpy as np




@numba.jit(nopython=True, fastmath=True)
def hessenberg(A, P):
    """
    Transformation inplace de A en sa forme de Hessenberg H
    P: matrice non-initialisée, contiendra la transformation unitaire
    """
    n = len(A)
    #stocker les vk:
    V = np.zeros((n,n), dtype=np.complex128)

    #reecriture de P en matrice identité:
    for i in range(n):
        for j in range(n):
            P[i,j] = 1. + 0.j if i == j else 0. + 0.j



    for k in range(n-2):
        e1 = np.zeros(n-k-1, dtype=np.complex128)
        e1[0] = 1.+0.j
        x = A[k+1:,k]
        val = np.abs(x[0])
        if val < 1e-12:
            val = 1
        coeff = (x[0]*complex_vector_norm(x)/val)
        v = coeff*e1 + x
        norm= complex_vector_norm(v)
        if norm != 0:
            v /= norm
        buff_out1 = np.zeros((n-k-1, n-k), dtype=np.complex128)
        buff_out2 = np.zeros((n, n-k-1), dtype=np.complex128)
        buff_row_matrix = np.zeros(n-k, dtype=np.complex128)
        buff_vec = np.zeros(n, dtype=np.complex128)
        V[k, k+1:] = v
        row_matrix_productinplace(np.conj(v), A[k+1:,k:], buff_row_matrix)
        outer_productinplacev2(v, buff_row_matrix, buff_out1)
        A[k+1:,k:] -= 2*buff_out1

        
        matrix_vector_productinplace(A[:, k+1:],v, buff_vec)
        outer_productinplacev2(buff_vec, np.conj(v), buff_out2)
        A[:, k+1:] -= 2*buff_out2
    
    #Obtenir P:
    for k in range(n-3, -1, -1):
        buff_out = np.zeros((n - k - 1,n - k - 1), dtype=np.complex128)
        buff_row = np.zeros(n - k - 1, dtype=np.complex128)

        row_matrix_productinplace(np.conj(V[k, k+1:]), P[k+1:,k+1:], buff_row)
        outer_productinplacev2(V[k, k+1:], buff_row, buff_out)
        P[k+1:,k+1:] -= 2*buff_out

    return 



@numba.jit(nopython=True, fastmath=True)
def step_qr(H, Q, m):
    """
    Réécriture inplace de H en le produit RQ avec conj(H) = QR

    H: matrice de Hessenberg
    Q: contiendra la transformation unitaire
    m: dimension de la matrice active
    """
    n = len(Q)
    rot_matrix = np.zeros((m-1, 2, 2), dtype=np.complex128)
    Gstar = np.zeros((2,2), dtype=np.complex128)

    for k in range(m-1):
            givens_complex(H[k:k+2, k], Gstar)
            rot_matrix[k] = np.conj(Gstar).T
            buff = np.zeros((2, n -k), dtype=np.complex128)
            multinplace(Gstar, H[k:k+2, k:], buff) 
            H[k:k+2, k:] = buff
    for k in range(m-1):
            G = rot_matrix[k]
            buff1 = np.zeros((k+2, 2), dtype=np.complex128)
            buff2 = np.zeros((n, 2), dtype=np.complex128)
            multinplace(H[:k+2, k:k+2], G, buff1)
            multinplace(Q[:,k:k+2], G, buff2)
            H[:k+2, k:k+2] = buff1
            Q[:,k:k+2] = buff2
    return 



#Les arguments supplémentaires sont pour les plots de convergence
@numba.jit(nopython=True, fastmath=True)
def step_qr_shift(H, Q, m, eps):
    """
    Calcul de la décomposition QR de H - sigma*I
    Retourne m_new: la nouvelle dimension de la matrice active
    """
    mu = 0. + 0.j
    a = H[m-2,m-2]
    b = H[m-2,m-1]
    c = H[m-1,m-2]
    d = H[m-1,m-1]
    trace = a + d

    
    delta = trace**2 - 4*(a*d - b*c)

    lambd1 = trace/2. + np.sqrt(delta)/2.
    lambd2 = trace/2. - np.sqrt(delta)/2.
    
    diff1 = np.abs(lambd1 - d)
    diff2 = np.abs(lambd2 - d)
    
    if diff1 < diff2:
        mu= lambd1
    else:  
        mu = lambd2
    for i in range(m):
        H[i,i] -= mu
    step_qr(H, Q, m)
    for i in range(m):
        H[i,i] += mu
    if np.abs(H[m-1, m-2]) < eps:
        return m-1
    return m


@numba.jit(nopython=True, fastmath=True)
def solve_qr(A, use_shifts, eps, max_iter):
    
    """
    Calcul des valeurs propres de A: Transformation in place de A en T (decomposition de Schur)
    use_shifts: booléen, utiliser ou non les shifts
    eps: critère d'arrêt
    max_iter: nombre maximal d'itérations

    retourne: U: contient la transformation unitaire de A
              k: nombre d'itérations qr nécessaires, -1 si on atteint max_iter
    """
    m = len(A)
    U = np.eye(m, dtype=np.complex128)
    k = 0
    #ite = np.zeros((max_iter, n), dtype=np.complex128) #pour les plots de convergence
    #Aold = np.copy(A) #pour plot la heat map
    #Aold_bonus = np.zeros((max_iter, n,n), dtype=np.complex128) #pour plot le bonus
    #k_defl = [] #pour plot convergence quadratique
    hessenberg(A, U)

    if use_shifts:
        while k < max_iter:
            #n_old = n #pour plot convergence quadratique
            m = step_qr_shift(A, U, m, eps)
            #if n != n_old:
                #k_defl.append(k)

            #ite[k,:] = np.diag(A)
            k+=1
            if  m <=1:
                break

    else:
        while k < max_iter and not check_convergence(A, eps,m):
            #Aold = np.copy(A)
            step_qr(A, U, m)
            #ite[k,:] = np.diag(A)
            k += 1
    if k == max_iter:
        k = -1
    return U, k#,Aold_bonus,  ite, k_defl, Aold #Choisir les arguments à renvoyer en fonction des plots à faire



#Fonctions auxiliaires:

#Produits matriciels compatibles avec numba:
@numba.jit(nopython=True, fastmath=True)
def row_matrix_productinplace(v, M, result):
    n = len(v)
    p = np.shape(M)[1]
    #result = np.zeros(p, dtype=np.complex128)
    for i in range(p):
        tmp = 0. + 0.j
        for j in range(n):
            tmp += v[j]*M[j,i]
        result[i] = tmp
    return


@numba.jit(nopython=True, fastmath=True)
def matrix_vector_productinplace(M, v, result):
    n = len(v)
    p = np.shape(M)[0]
    #result = np.zeros(p, dtype=np.complex128)
    for i in range(p):
        tmp = 0. + 0.j
        for j in range(n):
            tmp += M[i,j]*v[j]
        result[i] = tmp
    return 



@numba.jit(nopython=True, fastmath=True)
def outer_productinplacev2(v1, v2, M):
    n1 = len(v1)
    n2 = len(v2)
    #M = np.zeros((n1,n2), dtype=np.complex128)
    for i in range(n1):
        for j in range(n2):
            M[i,j] = v1[i]*v2[j]
    return 




@numba.jit(nopython=True, fastmath=True)
def complex_vector_norm(v):
    """
    Norme d'un vecteur complexe
    """
    res = 0.
    for i in range(len(v)):
        res += np.abs(v[i])**2

    return np.sqrt(res)


@numba.jit(nopython=True, fastmath=True)
def givens_complex(z, Gstar):
    """
    Calcul de la matrice de givens qui permet d'annuler la seconde composante d'un vecteur complexe
    """
    #Gstar = np.zeros((2,2), dtype=np.complex128)
    rho = np.sqrt(np.abs(z[0])**2 + np.abs(z[1])**2)
    cos = np.abs(z[0])/rho
    sin = np.abs(z[1])/rho
    phi1 = np.angle(z[0])
    phi2 = np.angle(z[1])
    Gstar[0,0] = cos
    Gstar[0, 1] = sin*np.exp(-(1j*phi2-1j*phi1))
    Gstar[1,0] = -sin*np.exp((1j*phi2-1j*phi1))
    Gstar[1,1] = cos


    return #np.conj(Gstar).T, Gstar


@numba.jit(nopython=True, fastmath=True)
def check_convergence(A, eps, n):
    """
    Check que les éléments sur la sous-diagonale sont à 0
    """
    for i in range(n-1):
        if np.abs(A[i+1,i]) > eps:
            return False
    return True


@numba.jit(nopython=True, fastmath=True, cache=True)
def multinplace(A, B, C):
    #A: m*n
    #B: n*p
    m,n  = np.shape(A)
    p = np.shape(B)[1]

    for i in range(m):
        for j in range(p):
            tmp = 0. + 0.j
            for k in range(n):
                tmp += A[i,k]*B[k,j]
            C[i,j] = tmp
                
    return   
    



