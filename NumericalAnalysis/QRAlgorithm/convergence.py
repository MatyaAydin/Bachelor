import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from devoir3 import *
from time import perf_counter as clock



"""
Remarque générale pour les plots: veillez à bien retourner les bons arguments dans solve_qr et à uncomment les lignes qui leur correspondent (et attention à l'ordre de retour)
Merci :)
"""
n = 7
nb_plot = 0 #A changer pour plot
eps = 1e-12
max_iter = 10000000




for _ in range(nb_plot):
    A = np.random.rand(n,n).astype(np.complex128) #+ 1j*np.random.rand(n,n).astype(np.complex128)
    #Si on veut A ssdp:
    A = A @ A.T

    eig = np.sort(scipy.linalg.eigvals(A))
    Abis = np.copy(A)


    U,k, ite, Aold, _ = solve_qr(A, False, eps, max_iter)
    U_shifted,k_shifted, ite_shifted, Aold_shifted, k_idx  = solve_qr(Abis, True, eps, max_iter)
    diff = np.zeros(k)
    diff_shifted = np.zeros(k_shifted)

    if k == -1:
        k = max_iter
    



    #Plot de l'erreur:
    
    #Preprocessing:
    #Sert à rien si je plot que la norme
    for i in range(k):
        ite[i] = np.sort(ite[i])
        ite_shifted[i] = np.sort(ite_shifted[i])


    diff = np.zeros(k)
    diff_shifted = np.zeros(k_shifted)

    for i in range(k):
        diff[i] = np.linalg.norm(ite[i] - eig)	
    for i in range(k_shifted):
        diff_shifted[i] = np.linalg.norm(ite_shifted[i] - eig)
    


    #Plot sans shift:
    big_O_lin = [1e-3*10**(-i) for i in range(k)]
    plt.plot(range(k), diff, label="Error without shift")
    #plt.plot(range(k_shifted), diff_shifted, label="Error with shift")
    plt.plot(range(k), big_O_lin, label="$\mathcal{O}(10^{-k})$", c="black", linestyle=":")
    


    
    plt.yscale("log")
    plt.xlabel("k [-]")
    plt.xticks(range(k))
    plt.ylabel("$||diag(A_k) - \lambda(A)||$ [-]")
    #plt.ylim(eps/10,1) #tres moche avec :(
    plt.title("Convergence of the QR algorithm for $\lambda \in \mathbb{R}$")
    plt.grid(True, axis='both', linestyle='--')
    plt.legend(loc='upper right')
    #plt.savefig("convergenceAsdp.pdf")
    plt.show()
    






    #Plot avec shift:

    defl_idx = [diff_shifted[i] for i in k_idx]

    big_O_quadratic = [5*10**(-2*i) for i in range(k_shifted)]
    plt.plot(range(k_shifted), diff_shifted, label="Error with shift")
    plt.plot(range(k_shifted), big_O_quadratic, label="$\mathcal{O}(10^{-2k})$", c="black", linestyle=":")
    plt.scatter(k_idx, defl_idx, c="red", s=20, label="Deflation index")

    plt.yscale("log")
    plt.xlabel("k [-]")
    plt.xticks(range(k_shifted))
    plt.ylabel("$||diag(A_k) - \lambda(A)||$ [-]")
    #plt.ylim(eps/10,1) #tres moche avec :(
    plt.title("Convergence of the QR algorithm for $\lambda \in \mathbb{R}$")
    plt.grid(True, axis='both', linestyle='--')
    plt.legend(loc='upper right')
    #plt.savefig("convergenceAsdpwithshift.pdf")
    plt.show()
    





#Heatmap sans shift: il faut modifier solve_qr pour retourner le dernier A_k avant convergence
#Générer une matrice avec des lambdas au choix: 
#source: https://stackoverflow.com/questions/49574212/random-positive-semi-definite-matrix-with-given-eigenvalues-and-eigenvectors
des = [5, 4,3,2,1]
n = len(des)
s = np.diag(des)
q, _ = scipy.linalg.qr(np.random.rand(n, n))
semidef = q.T @ s @ q
semidef = semidef.astype(np.complex128)

U,k, ite, Aold, _ = solve_qr(semidef, False, eps, max_iter)
heat = np.divide(np.abs(semidef), np.abs(Aold))
heat[2:, 0] = 0
heat[3:, 1] = 0
heat[4, 2] = 0
ax = sns.heatmap(heat, linewidth=0.5, vmin=0, vmax=1)
plt.title("|$\dfrac{A_k}{A_{k-1}}|$ elements without shift")
#plt.savefig("heatmap_no_shift.pdf")
plt.show()




#Comparaison de convergence avec et sans shift:

nb_size = 0 #A changer pour plot
size = range(10,nb_size*10 + 10,10)
k_noshift = np.zeros(nb_size)
k_shift = np.zeros(nb_size)
i = 0
for n in size:
    A = np.random.rand(n,n).astype(np.complex128)
    A = A.T @ A
    A_cpy = np.copy(A)

    U,k = solve_qr(A, False, eps, max_iter)
    U_shifted,k_shifted = solve_qr(A_cpy, True, eps, max_iter)
    if k == -1:
        k = max_iter
    if k_shifted == -1:
        k_shifted = max_iter
    k_noshift[i] = k
    k_shift[i] = k_shifted
    print(i)
    i+=1


On = [1e3*i for i in size]

plt.xscale("log")
plt.yscale("log")
plt.grid(True, axis='both', linestyle='--')
plt.plot(size, k_noshift, label="Without shift")
plt.plot(size, k_shift, label="With shift")
plt.plot(size, On, label="$\mathcal{O}(n)$", c="black", linestyle=":")
plt.xlabel("$n$ [-]")
plt.ylabel("$k$ [-]")
plt.title("Number of iterations $k$ for $\lambda \in \mathbb{R}$," + r'$A \in \mathbb{R}^{n \times n}$')
plt.legend(loc = 'upper right')
#plt.savefig("convergence_comparison.pdf")
plt.show()
    





###Complexité###

#pour que numba précompile les fonctions et que le premier temps mesuré soit pas erroné:
U,k = solve_qr(np.eye(4, dtype= np.complex128), False, eps, max_iter)
U,k = solve_qr(np.eye(4, dtype= np.complex128), True, eps, max_iter)

nb_size = 0 #A changer pour plot
time_schur = np.zeros(nb_size)
time_solve = np.zeros(nb_size)
time_solve_shift = np.zeros(nb_size)
i = 0
#Sans shift vs scipy.schur:
size = range(10,nb_size*10 + 10,10)
for n in size:
    A = np.random.rand(n,n).astype(np.complex128)
    A = A @ A.T
    A_cpy = np.copy(A)
    start_scipy = clock()
    T,Z = scipy.linalg.schur(A)
    end_scipy = clock()


    start_solve = clock()
    U,k = solve_qr(A, False, eps, max_iter)
    end_solve = clock()

    start_shift = clock()
    U_shifted,k_shifted = solve_qr(A_cpy, True, eps, max_iter)
    end_shift = clock()

    time_schur[i] = end_scipy - start_scipy
    time_solve[i] = end_solve - start_solve
    time_solve_shift[i] = end_shift - start_shift
    i+=1
    print(i)

big_O_cube = [1e-4*i**3 for i in size]


plt.xscale("log")
plt.yscale("log")
plt.plot(size, time_schur, label="scipy.schur")
plt.plot(size, time_solve, label="QR algorithm without shift")
plt.plot(size, time_solve_shift, label="QR algorithm with shift")
plt.plot(size, big_O_cube, label="$\mathcal{O}(n^3)$", c="black", linestyle=":")

plt.xlabel("n [-]")
plt.ylabel("Time [s]")
plt.title("Scipy vs solve_qr")
plt.grid(True, axis='both', linestyle='--')
plt.legend(loc = 'upper right')
#plt.savefig("complexity.pdf")
plt.show()







#Bonus:
#Il faut modifier solve_qr pour conserver chaque matrice A_k
"""
des = [4,3, 1-2j, 1+2j]
n = len(des)
s = np.diag(des)
q, _ = scipy.linalg.qr(np.random.rand(n, n))
semidef = q.T @ s @ q
semidef = semidef.astype(np.complex128)
max_iter = 100
U,k, ite, Aold, kdefl = solve_qr(semidef, False, 1e-12, max_iter)


Aold_real = np.real(Aold)
Aold_imag = np.imag(Aold)
plt.scatter(Aold_real[:,3,2], Aold_imag[:,3,2])
plt.title("$a_{34}$'s trajectory in $\mathbb{C}$ without shift")
plt.xlabel("$\Re(a_{34})$")
plt.ylabel("$\Im(a_{34})$")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axis('scaled')
plt.savefig("bonus.pdf")
plt.show()
"""


