"""
LINMA1731's project - Part 2 - Group 18
Authors:
    -Ari Prezerowitz
    -Matya Aydin
"""


#Load packages
import numpy as np
import matplotlib.pyplot as plt
import random


#Load data
observations = np.loadtxt('data/Observations.txt')
true_data = np.loadtxt('data/True_data.txt')

trajectory_xy = np.zeros((60, 2))
trajectory_v = np.zeros((60, 2))

np.copyto(trajectory_xy, true_data[:, :2])
np.copyto(trajectory_v, true_data[:, 2:])


#to generalise result:
nb_experiment = 25 #Can be increased for kalman but will take a lot of time for the particle filter

#Parameters
t_f = 60 #Number of observations
dt = 0.1
g  = 9.81
mu = 1
sigma = 1
#Noise on observations
R = np.eye(2)



#Matrices definition

#State transition matrix
A = np.eye(4)
A[0, 2] = dt
A[1, 3] = dt

#Disturbance on state matrix
G = np.zeros((4, 2))
G[0,0] = 0.5*dt**2
G[1,1] = 0.5*dt**2
G[2,0] = dt
G[3,1] = dt

#Input matrix
D = np.zeros(4)
D[1] = 0.5*g*dt**2
D[3] = g*dt

#Output matrix
C = np.zeros((2, 4))
C[0, 0] = 1
C[1, 1] = 1




#Covariance between states matrix
P_pos = 10
P_vel = 10
P = np.diag([P_pos, P_pos, P_vel, P_vel])


Q = (np.exp(sigma) - 1) *np.exp(2*mu + sigma) * np.dot(G, G.T)

#As explained, we need to add a shift to the prediction step since the noise is a lognormal with non-zero mean instead of WGN
shift = np.exp(mu + sigma/2)*np.ones(2)







def kalman_filter(A, D, G, C, P, shift, nb_experiment):
    filtered_values = np.zeros((len(observations), 2))
    #Initial state
    x = np.zeros(4)


    for j in range(nb_experiment):

        #Randomly initialises the initial state
        x[0:2] = np.random.normal(10, 50, 2)
        x[2:] = np.random.normal(10, 10, 2)
        #We initialize M with the covariance matrix between states P
        M = P

        #Kalman filter
        for i in range(t_f):
            #prediction without new measurement:
            x = np.dot(A, x) - D + np.dot(G, shift)
            #update M:
            M = np.dot(np.dot(A, M), A.T) + Q
            #Kalman gain:
            K = np.dot(np.dot(M, C.T), np.linalg.inv(np.dot(np.dot(C, M), C.T) + R))
            #correction:
            x = x + np.dot(K, (observations[i] - np.dot(C, x)))
            #calculates the MMSE x[n|n]:
            M = M - np.dot(np.dot(K, C), M)
            filtered_values[i] += np.dot(C, x)/nb_experiment

    
    return filtered_values






#Kalman filter plot

filtered_values = kalman_filter(A, D, G, C, P, shift, nb_experiment)


#MSE
error = np.zeros(t_f)
MSE = 0
for j in range(t_f):
    MSE_ite = (np.linalg.norm(true_data[j,:2] - filtered_values[j])**2)/t_f
    MSE += MSE_ite
    error[j] =MSE_ite

MSE_plot = round(MSE, 2)


plt.scatter(observations[:, 0], observations[:, 1], label='Noisy observations')
#Uncomment to check the answer
#plt.scatter(true_data[:, 0], true_data[:, 1], label='True position')
plt.scatter(filtered_values[:, 0], filtered_values[:, 1], label='Filtered position')

#Adds ground and target (small esthetic details :) )
plt.plot((-10, 0), (45, 45), color='saddlebrown', linestyle='--')
plt.plot((0, 0), (0, 45), color='saddlebrown', linestyle='--')
plt.plot((70, 70), (0, 10), color='saddlebrown', linestyle='--')
plt.plot((70, 90), (10, 10), color='saddlebrown', linestyle='--')
plt.scatter(80, 10, color='red',s = 100, marker = 'x')

plt.legend(loc = 'upper right')
plt.title(f'Kalman filter for {nb_experiment} experiments')
plt.xlabel('x-coordinate $[m]$')
plt.ylabel('y-coordinate $[m]$')
#plt.savefig('Kalman_filter.pdf')
plt.show()


#Error plot (MSE)
plt.plot(range(t_f), error)
plt.text(0, 0.0, f'Mean MSE = {MSE_plot}', fontsize = 10)
plt.title(f'Error of the kalman filter for {nb_experiment} experiments')
plt.xlabel('nth step [-]')
plt.ylabel('Error on position $[m^2]$')
#plt.savefig("error_kalman.pdf")
plt.show()







###Particle filter###
d_s = 4         # dimension of state space
d_x = 2         # dimension of output space


#Covariance of initial states
cov_init = np.identity(4)
cov_init[0,0] = cov_init[1,1] = 50
cov_init[2,2] = cov_init[3,3] = 10

mu_s = 10        
C_s = cov_init  
mu_w = np.zeros(d_x)



out_noise_pdf = lambda w: 1/np.sqrt((2*np.pi)**d_x*np.abs(np.linalg.det(R))) * np.exp(-.5*(w-mu_w)@np.linalg.inv(R)@(w-mu_w))  # pdf of the output noise w_t





def ParticleFilter(n_part, tol):
    
    # Number of particles. Sugg: 1e2
    particles = np.zeros((d_s,n_part,t_f +1))   
    new_particles = np.zeros((d_s,n_part,t_f +1))  # to store the updated particles 
    weights = 1/n_part*np.ones(n_part)            # Initialize weights

    
    #Generate particles from initial conditions
    for i in range(n_part):
        particles[0,i,0] = np.random.normal(mu_s,C_s[0,0])
        particles[1,i,0] = np.random.normal(mu_s,C_s[1,1])
        particles[2,i,0] = np.random.normal(mu_s,C_s[2,2])
        particles[3,i,0] = np.random.normal(mu_s,C_s[3,3])

    #Output
    predictions = np.zeros((t_f, 2))

    #iterates through time to predict each state
    for n in range(t_f):



        for i in range(n_part):
            new_particles[:, i, n+1] = A @ particles[:, i, n] + G @ np.random.lognormal(1, 1, 2) - D
        

        # Weights computation
        for i in range(n_part):
            z = out_noise_pdf(observations[n] - C@(new_particles[:,i,n+1]))
            weights[i] = z
        if np.sum(weights) > tol:
            weights = weights/np.sum(weights)
        else:
            #if the weights are too small, we reinitialize them
            weights = np.ones(n_part)/n_part


        #Prediction
        prediction = np.zeros(2)

        for i in range(n_part):
            prediction += weights[i] * new_particles[:2,i,n+1]
        predictions[n] = prediction


        #Resampling
        ind_sample = random.choices(population=np.arange(n_part), weights=weights, k=n_part)
        for i in range(n_part):

            particles[:, i, n+1] = new_particles[:, ind_sample[i], n+1]
    
    return predictions
    




### To generalize results:
#Compare simple average with average weighted by the MSE
pf_prediction = np.zeros((t_f,2))
pf_prediction_base = np.zeros((t_f,2))
MSE_exp = np.zeros(nb_experiment)
for i in range(nb_experiment):
    predcition_i=ParticleFilter(1000, 1e-15)
    for j in range(1,60-2):
        MSE_exp[i] += np.linalg.norm(true_data[j,:2]-predcition_i[j])**2/t_f
    MSE_exp[i] = 1/MSE_exp[i]
    pf_prediction += MSE_exp[i]*predcition_i
    pf_prediction_base += predcition_i/nb_experiment

#pf_prediction /= nb_experiment
pf_prediction /= np.sum(MSE_exp)






MSE=0
error_particle = np.zeros(t_f - 2 - 1 +1)
error_particle_base = np.zeros(t_f - 2 - 1 +1)
for i in range(1,60-2):
    MSE_step = np.linalg.norm(true_data[i,:2]-pf_prediction[i])**2
    #MSE+= MSE_step
    error_particle[i] = MSE_step
    MSE_step_base = np.linalg.norm(true_data[i,:2]-pf_prediction_base[i])**2
    MSE += MSE_step_base
    error_particle_base[i] = MSE_step_base
MSE=MSE/t_f  
MSE_plot = np.round(MSE,2)


#Particle filter plot:
plt.figure()
plt.scatter(true_data[:,0], true_data[:,1], marker='o', color="blue", label="true position")
plt.scatter(observations[:,0], observations[:,1], marker='s', color="orange", label="observations")
plt.scatter(pf_prediction[:,0], pf_prediction[:,1], marker='^', color="green", label = "estimation")

#Adds ground and target (small esthetic details :) )
plt.plot((-10, 0), (45, 45), color='saddlebrown', linestyle='--')
plt.plot((0, 0), (0, 45), color='saddlebrown', linestyle='--')
plt.plot((70, 70), (0, 10), color='saddlebrown', linestyle='--')
plt.plot((70, 90), (10, 10), color='saddlebrown', linestyle='--')
plt.scatter(80, 10, color='red',s = 100, marker = 'x')

plt.legend()
plt.grid()
plt.title(f"Particle Filter repeated {nb_experiment} times")
plt.xlabel("x-coordinate [m]")
plt.ylabel("y-coordinate [m]")
#plt.savefig("particle_filter.pdf")
plt.show()


#Error plot (MSE)
#plt.plot(range(1,t_f - 2 +1), error_particle, label="MSE-weighted average")
plt.plot(range(1,t_f - 2 +1), error_particle_base, label = "Average")
plt.text(0, 0.1, f'Mean MSE = {MSE_plot}', fontsize = 10)
plt.title(f'Error of the particle filter for {nb_experiment} experiments')
plt.xlabel('nth step [-]')
plt.ylabel('Error on position $[m^2]$')
plt.legend(loc = "upper left")
#plt.savefig("error_particle.pdf")
plt.show()


#MSE wrt number of particles plot

nb_particles = range(1000, 11000, 1000)
err_particles = np.zeros(len(nb_particles))

for j in range(len(nb_particles)):
    n_part = nb_particles[j]
    pf_prediction = np.zeros((t_f, 2))

    for i in range(nb_experiment):
        predcition_i=ParticleFilter(n_part, 1e-15)
        pf_prediction +=predcition_i

    pf_prediction /= nb_experiment

    MSE=0
    for i in range(1,t_f-2):
        MSE_step = MSE_step = np.linalg.norm(true_data[i,:2]-pf_prediction[i])**2
        MSE += MSE_step
    MSE=MSE/t_f
    err_particles[j] = MSE
    print("done")

plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", ls="--")

plt.plot(nb_particles, err_particles)
plt.xlabel("$N_p$ [-]")
plt.ylabel("Mean MSE $[m^2]$")
plt.title(f"MSE of the particle filter wrt $N_p$ repeated {nb_experiment} times")
#plt.savefig("error_particle_nbparticle.pdf")
plt.show()



#MSE wrt weights tolerance plot

weights_tol = np.logspace(-16, -9, 10)
err_particles = np.zeros(len(weights_tol))

for j in range(len(weights_tol)):
    tol = weights_tol[j]
    pf_prediction = np.zeros((t_f,2))

    for i in range(nb_experiment):
        predcition_i=ParticleFilter(1000, tol)
        pf_prediction +=predcition_i

    pf_prediction /= nb_experiment

    MSE=0
    for i in range(1,t_f-2):
        MSE_step = MSE_step = np.linalg.norm(true_data[i,:2]-pf_prediction[i])**2
        MSE += MSE_step
    MSE=MSE/t_f
    err_particles[j] = MSE
    print("done")

plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", ls="--")

plt.plot(weights_tol, err_particles)
plt.xlabel("$\epsilon$ [-]")
plt.ylabel("Mean MSE $[m^2]$")
plt.title(f"MSE of the particle filter wrt weight tolerance repeated {nb_experiment} times")
#plt.savefig("error_particle_tolerance.pdf")
plt.show()







#Resistance to noise plot: involves both filter
sigma_obs = np.arange(1, 11)
sigma_obs = sigma_obs**2
MSE_sigma_kalman = np.zeros(len(sigma_obs))
MSE_sigma_particle = np.zeros(len(sigma_obs))

for i in range(len(sigma_obs)):
    R = np.eye(2)*sigma_obs[i]

    filtered_values = kalman_filter(A, D, G, C, P, shift, nb_experiment)
    MSE_kalman = 0
    for j in range(t_f):
        MSE_kalman += (np.linalg.norm(true_data[j, :2] - filtered_values[j])**2)/t_f
    MSE_sigma_kalman[i] = MSE_kalman
    print(MSE_kalman)


    pf_prediction = np.zeros((t_f,2))
    for _ in range(nb_experiment):
        predcition_i=ParticleFilter(1000, 1e-15)
        pf_prediction +=predcition_i/nb_experiment

    MSE_particle = 0
    for j in range(1,t_f-2):
        MSE_step =np.linalg.norm(true_data[j,:2]-pf_prediction)**2
        MSE_particle += MSE_step

    MSE_particle /= t_f
    MSE_sigma_particle[i] = MSE_particle
    


    print(MSE_particle, "\n")



plt.xscale("log")
plt.yscale("log")

plt.grid(True, which="both", ls="--")

plt.plot(sigma_obs, MSE_sigma_kalman, "o-", label = "Kalman filter")
plt.plot(sigma_obs, MSE_sigma_particle, "o-", label="Particle filter")
plt.title(f"MSE wrt observation noise for {nb_experiment} experiments")
plt.xlabel("Observation noise $\sigma_{v}^2$")
plt.ylabel("Mean MSE $[m^2]$")
plt.legend(loc = "upper left")
#plt.savefig("MSE_observation_noise.pdf")
plt.show()

