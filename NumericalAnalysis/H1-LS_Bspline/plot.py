import matplotlib.pyplot as plt
import numpy as np
from devoir1 import *

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

def chordal_parametrization(P):
    """
    Computes the m parameters for the m points using the chordal parametrization
    @param: -m: number of points to compute
            -fx: x(t)
            -fy: y(t)

    @return: T: numpy array containing the m parameters
    """
    m = len(P)
    T = np.zeros((m))
    T[0] = 0
    T[-1] = 1
    D = 0
    for i in range(1, m):
       D += norm(P[i] - P[i - 1])

    for i in range(1, m):
       T[i] = T[i-1] + norm(P[i] - P[i - 1])/D
    
    return T


#NE PAS PRENDRE N = 3 SINON DIVZERO ICI
def control_points(n, t):
    """
    4 premiers a 0
    4 derniers a 1
    """
    m = len(t)
    size = n + 4
    T = np.zeros((size))
    d = m/(n - 3)
    for i in range(1, 5):
        T[size - i] = 1
    
    for j in range(1, size - 6):
        """
        reste n + 4 - 8 = n - 4 point a fixer: indice[4 -->size-5]
        """
        i = int(np.floor(j*d)) - 1
        alpha = j*d - i
        T[j + 3] = (1 - alpha)*t[i - 1] + alpha*t[i]



    return T



def b(t,T,i,p):
  """
  Recursive B-spline definition found in the numerical methods LEPL1104 course.
  @param: -t: coordinate at which we evaluate the spline
          -T: parameters knots
          -i: [1; len(T)-1] ith term in the recursive definition
          -p: degree of the spline

    @return u: b_i(t)
  """


  if p == 0:
    return (T[i] < t)*(t < T[i+1]) #init: < dans deuxieme ()
  
  
  else:
    u  = 0.0 if T[i+p ]  == T[i]   else (t-T[i])/(T[i+p]- T[i]) * b(t,T,i,p-1)
    u += 0.0 if T[i+p+1] == T[i+1] else (T[i+p+1]-t)/(T[i+p+1]-T[i+1]) * b(t,T,i+1,p-1)
    return u
  
def generate_points(m, start, end, fx, fy):
   """
   courbe parametree (x(t), y(t))
   """
   dom = np.linspace(start, end, m)
   P = np.zeros((m,2))
   for i in range(m):
      P[i,:] = np.array([fx(dom[i]), fy(dom[i])])

   return P
 

#Fonctions pour les dessins
def sin(x):
   return np.sin(x)
def cos(x):
   return np.cos(x)
def identity(x):
   return x
def square(x):
   return x**2
def spirale_x(x):
   return x*cos(x)
def spirale_y(x):
   return x*sin(x)
def heart_x(x):
   return 16*(sin(x))**3
def heart_y(x):
   return 13*cos(x) - 5*cos(2*x) - 2*cos(3*x) - cos(4*x)
def inf_y(x):
   return np.cos(x)*np.sin(x)



#Fonction qui automatise les plots:
def do_everything(m_list, n_list, start, end, fx, fy, nb_spline, name):
   """
   @param:-m: nombre de points a approximer
         -n: nombre de points de controle
         -start: début du domaine de la courbe parametree
         -end: fin du domaine de la courbe parametree
         -fx: coordonnee x de la courbe parametree
         -fy: coordonnee y de la courbe parametree
         -nb_bspline: nombre de points pour rendre l'approximation plus lisse
         -name: nom de la figure pour sauvegarder le plot
   """

   size = len(m_list)
   fig, axes = plt.subplots(ncols=size, figsize=(12, 5))
   for k in range(size):
      m = m_list[k]
      n = n_list[k]
      #Parametrisation:
      P = generate_points(m, start, end, fx, fy)
      t = chordal_parametrization(P) 
      T = control_points(n, t)


      #Initialisation du systeme:
      A = np.zeros((m, n))

      for i in range(m):
         for j in range(n):
            A[i][j] = b(t[i], T, j, 2)

      #On vérifie bien qu'on a une matrice quasi diagonale:      
      #plt.spy(A)
            
      A[0][0] = 1
      A[-1][-1] = 1

      #Résolution du système:
      X = lstsq(A, P)


      #Jolis dessins :)
      nb_spline = 200
      b_x = np.zeros(nb_spline)
      b_y = np.zeros(nb_spline)
      t_spline = np.linspace(0, 1, nb_spline, endpoint=False)


      for i in range(n):
         for j in range(nb_spline):
            b_x[j] += X[i,0]*b(t_spline[j], T, i,2)
            b_y[j] += X[i,1]*b(t_spline[j], T, i,2)


      axes[k].scatter(P[:,0], P[:,1], s=8, label="Approximated knots")
      axes[k].scatter(X[:,0], X[:,1], s=30, marker="d", label="Control knots")
      axes[k].plot(b_x[1:], b_y[1:], label="Curve", c="green")
      axes[k].set_title("m={}, n ={}".format(m, n))
      axes[k].legend(loc='upper right')
      axes[k].grid(True)
      axes[k].set_aspect('equal', adjustable='box')

   fig.suptitle('Least squares approximation using B-splines', fontsize=12)
   plt.tight_layout()
   #plt.savefig("3cases{}.pdf".format(name))
   plt.show()



#Dimensions: les 3 cas m>>n, m>n et m = n
n = [15, 100, 200]
m = [200, 200, 200]

#Plot du cercle:
do_everything(m, n, 0, 2*np.pi, cos, sin, 200, "Circle")


#Plot du coeur:
do_everything(m, n, 0, 2*np.pi, heart_x, heart_y, 200, "Heart")

#Plot de la spirale:
do_everything(m, n, 0, 6*np.pi, spirale_x, spirale_y, 200, "Spiral")

#Plot de l'infini:
do_everything(m, n, -0.5*np.pi, 3*np.pi, cos, inf_y, 200, "infinity")
