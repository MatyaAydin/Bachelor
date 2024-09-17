"""
Conditionnement des moindres carrés en utilisant les formules du livre de référence
Code partiellement inspiré du code "condition.py" fourni sur moodle 
"""

import numpy as np
import matplotlib.pyplot as plt



#dimension du problème
m = 2

#résolution des moindres carrés
A = np.random.randn(m,m)
B = np.random.randn(m)
X = np.linalg.lstsq(A, B, rcond=None)[0]

#Données utiles pour le conditionnement
kappa = np.linalg.cond(A)
y = A @ X
val = min(np.linalg.norm(y) / np.linalg.norm(B), 1)
theta = np.arccos(val)
eta = np.linalg.norm(A) * np.linalg.norm(X) / np.linalg.norm(y)



#Perturbation sur A:
p = 2000
delta = np.zeros((p,2))
lstq_bound_A = kappa + kappa**2 * np.tan(theta)/eta
for i in range(p):
    Ap = A + 1e-10 * np.random.randn(m,m)
    Xp = np.linalg.lstsq(Ap, B, rcond=None)[0]
    delta[i,:] = ((Xp - X) / np.linalg.norm(X)) / (np.linalg.norm(Ap - A) / np.linalg.norm(A))

fig,ax = plt.subplots()
ax.scatter(delta[:,0], delta[:,1], label= r' $\dfrac{ \dfrac{||\delta x ||}{||x||}}{\dfrac{||\delta A ||}{||A||}}$')
circle = plt.Circle((0.0,0.0), lstq_bound_A, fill=False, label = r'$\dfrac{\kappa (A)^2  tan(\theta )}{\eta} + \kappa (A)$', color="red", linestyle="--")
ax.add_patch(circle)

plt.legend(loc='upper right')
plt.title("Conditionnement des moindres carrés: perturbation sur $A$, $m = 2$")
plt.xlabel("$\delta x_0 [-]$")
plt.ylabel("$\delta x_1 [-]$")
plt.grid(True, linestyle='--')
plt.savefig("cond_lstsq_A.pdf")
plt.show()




#Perturbation sur b:
lstq_bound_B = kappa / (eta*np.cos(theta))

for i in range(p):
    Bp = B + 1e-10 * np.random.randn(m)
    Xp = np.linalg.lstsq(A, Bp, rcond=None)[0]
    delta[i,:] = ((Xp - X) / np.linalg.norm(X)) / (np.linalg.norm(Bp - B) / np.linalg.norm(B))

fig,ax = plt.subplots()
ax.scatter(delta[:,0], delta[:,1], label= r' $\dfrac{ \dfrac{||\delta x ||}{||x||}}{\dfrac{||\delta b ||}{||b||}}$')
circle = plt.Circle((0.0,0.0), lstq_bound_B, fill=False, label=r' $\dfrac{\kappa (A)}{\eta cos(\theta)}$', color="red", linestyle="--")
ax.add_patch(circle)

plt.legend(loc='upper right')
plt.title("Conditionnement des moindres carrés: perturbation sur $b$, $m = 2$")
plt.xlabel("$\delta x_0 [-]$")
plt.ylabel("$\delta x_1 [-]$")
plt.grid(True, linestyle='--')
plt.savefig("cond_lstsq_B.pdf")
plt.show()



#Plus haute dimension:
dim = 1000

#perturbation sur A:


delta_dim_A = np.zeros(dim)
bound_dim_A = np.zeros(dim)


for i in range(1, dim+1):
    A = np.random.randn(i, i)
    B = np.random.randn(i)
    X = np.linalg.lstsq(A, B, rcond=None)[0]

    kappa = np.linalg.cond(A)
    y = A @ X
    val = min(np.linalg.norm(y) / np.linalg.norm(B), 1)
    theta = np.arccos(val)
    eta = np.linalg.norm(A) * np.linalg.norm(X) / np.linalg.norm(y)
    bound_dim_A[i-1] = kappa + kappa**2 * np.tan(theta)/eta

    Ap = A + 1e-10 * np.random.randn(i,i)
    Xp = np.linalg.lstsq(Ap, B, rcond=None)[0]
    delta_dim_A[i-1] = np.linalg.norm(Xp - X) / np.linalg.norm(X) / (np.linalg.norm(Ap - A) / np.linalg.norm(A))


plt.plot(range(dim), delta_dim_A, label="Norme de la perturbation")  
plt.plot(range(dim), bound_dim_A, label="Nombre de conditionnement")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, linestyle='--')
plt.legend(loc = "upper right")
plt.title("Perturbation sur $A$ en fonction de la dimension")
plt.xlabel("Dimension [-]")
plt.ylabel("Norme [-]")
plt.savefig("cond_lstsq_A_dim.pdf")
plt.show()



#perturbation sur b:
delta_dim_B = np.zeros(dim)
bound_dim_B = np.zeros(dim)


for i in range(1, dim+1):
    A = np.random.randn(i, i)
    B = np.random.randn(i)
    X = np.linalg.lstsq(A, B, rcond=None)[0]

    kappa = np.linalg.cond(A)
    y = A @ X
    val = min(np.linalg.norm(y) / np.linalg.norm(B), 1)
    theta = np.arccos(val)
    eta = np.linalg.norm(A) * np.linalg.norm(X) / np.linalg.norm(y)
    bound_dim_B[i-1] = kappa / (eta*np.cos(theta))

    Bp = A + 1e-10 * np.random.randn(i,i)
    Xp = np.linalg.lstsq(A, Bp, rcond=None)[0]
    delta_dim_B[i-1] = np.linalg.norm(Xp - X) / np.linalg.norm(X) / (np.linalg.norm(Bp - B) / np.linalg.norm(B))


plt.plot(range(dim), delta_dim_B, label="Norme de la perturbation")  
plt.plot(range(dim), bound_dim_B, label="Nombre de conditionnement")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, linestyle='--')
plt.legend(loc = "upper right")
plt.title("Perturbation sur $b$ en fonction de la dimension")
plt.xlabel("Dimension [-]")
plt.ylabel("Norme [-]")
plt.savefig("cond_lstsq_B_dim.pdf")
plt.show()