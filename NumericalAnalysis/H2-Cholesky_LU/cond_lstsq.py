import numpy as np
import matplotlib.pyplot as plt


#Code fortement inspiré de celui fourni lors du CM2 "condition.py"

m = 2

A = np.random.randn(m,m)
B = np.random.randn(m)
X = np.linalg.solve(A.T@A, A.T@B)
kappa = np.linalg.cond(A.T@A)



p = 2000
delta = np.zeros((p,2))

#perturbation sur A:
for k in range(p):
    Ap = A.T@A + 1e-10 * np.random.randn(m,m)
    xp = np.linalg.solve(Ap, A.T@B)
    delta[k,:] = ((xp - X) / np.linalg.norm(X)) / (np.linalg.norm(Ap - A.T@A) / np.linalg.norm(A.T@A))

fig,ax = plt.subplots()
ax.scatter(delta[:,0], delta[:,1], label= r' $\dfrac{ \dfrac{||\delta x ||}{||x||}}{\dfrac{||\delta A ||}{||A||}}$')
circle = plt.Circle((0.0,0.0), kappa, fill=False, label="$\kappa (A)$", color="red", linestyle="--")
ax.add_patch(circle)

plt.legend(loc='upper right')
plt.title("Conditionnement des moindres carrés: perturbation sur $A$")
plt.xlabel("$\delta x[0] [-]$")
plt.ylabel("$\delta x[1] [-]$")
plt.grid(True, linestyle='--')
plt.savefig("cond_lstsq_A.pdf")
plt.show()





#Perturbation sur b:
delta = np.zeros((p,2))
eta = np.linalg.norm(A) * np.linalg.norm(X) / np.linalg.norm(A @ X)
kappa_b = kappa / eta
for k in range(p):
    Bp = B + 1e-10 * np.random.randn(m)
    xp = np.linalg.solve(A.T @ A, A.T@Bp)
    delta[k,:] = ((xp - X) / np.linalg.norm(X)) / (np.linalg.norm(A.T @ Bp - A.T @ B) / np.linalg.norm(A.T @ B))

fig,ax = plt.subplots()
ax.scatter(delta[:,0], delta[:,1], label= r' $\dfrac{ \dfrac{||\delta x ||}{||x||}}{\dfrac{||\delta b ||}{||b||}}$')
circle = plt.Circle((0.0,0.0), kappa_b, fill=False, label=r' $\dfrac{\kappa (A)}{\eta}$', color="red", linestyle="--")
ax.add_patch(circle)

plt.legend(loc='upper right')
plt.title("Conditionnement des moindres carrés: perturbation sur $b$")
plt.xlabel("$\delta x[0] [-]$")
plt.ylabel("$\delta x[1] [-]$")
plt.grid(True, linestyle='--')
plt.savefig("cond_lstsq_B.pdf")
plt.show()