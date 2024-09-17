from lu import lu
from cholesky import cholesky
from time import perf_counter as clock
import numpy as np 
import scipy.linalg
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
import seaborn as sns





m = 5000
time_LU = np.zeros(m//10 +1, dtype="float64")
time_cholesky = np.zeros(m//10 +1, dtype="float64")
ratio = np.zeros(m//10 +1, dtype="float64")

#Echauffer Numba:
L,U = lu(np.random.rand(3,3))
R = cholesky(np.random.rand(3,3))


#Calcul des temps d'exécution:
for i in range(1,m+2, 10):
    A = np.random.rand(i, i)
    spd = A.T @ A
    spd_chol = np.copy(spd)

    start = clock()
    L = lu(spd)
    end = clock()
    time_LU[(i-1)//10] = end-start

    start_cho = clock()
    chol = cholesky(spd_chol)
    end_cho = clock()
    time_cholesky[(i-1)//10] = end_cho - start_cho

    ratio[(i-1)//10] = time_LU[(i-1)//10] / time_cholesky[(i-1)//10]

    print("ite {}, ratio = {}".format(i, ratio[(i-1)//10]))


#Quelques stats:
print("-------------------------------------------------------")
print("Statistics:")
print("Mean = {}".format(np.mean(ratio)))
print("Std = {}".format(np.std(ratio)))
print("Median = {}".format(np.median(ratio)))




abscisse = range(1,m+2, 10)
ref = 2*np.ones(m//10 +1)
plt.plot(abscisse[4:], ratio[4:], label= r' $\dfrac{LU}{Cholesky}$', color="green")
plt.plot(abscisse, ref, label="Résultat théorique", color="blue", linestyle="--")

plt.title("Ratios des temps d'exécution en fonction de la dimension de A")
plt.xlabel("Dimensions de A [-]")
plt.ylabel("Ratio [-]")
plt.legend(loc='upper right')
plt.xlim((10, m))
plt.ylim((0, 3))
plt.savefig("Gain.pdf")
plt.show()


#Plot loglog:

big_O_cube = [1e-7*i**3 for i in abscisse]

#Les données en 1 sont fausses car le temps d'exécution est trop faible pour être mesuré correctement
plt.plot(abscisse[1:], time_LU[1:], label="LU")
plt.plot(abscisse[1:], time_cholesky[1:], label="Cholesky")
plt.plot(abscisse, big_O_cube, label="$\mathcal{O}(n^3)$", c="black", linestyle=":")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Dimensions de A [-]")
plt.ylabel("Temps d'exécution $[s]$")
plt.xlim((10, m))
plt.title("Temps d'exécution des décompositions LU et Cholesky")
plt.grid(True, axis='both', linestyle='--')
plt.legend(loc='upper right')
plt.savefig("gain_loglog.pdf")
plt.show()




#approche bar chart:
ratio_bar = np.round(ratio)
values, counts = np.unique(ratio_bar, return_counts=True)
sns.set()
bar = sns.barplot(x=values,y=counts, color='steelblue')
bar.set(xlim=(0,10))
plt.xlabel("Ratio [-]")
plt.ylabel("Occurence [-]")
plt.title("Occurence des ratios  $\dfrac{LU}{Cholesky}$ arrondis à l'unité")
plt.savefig("Gain_Bar.pdf")
plt.show()



