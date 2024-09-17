import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


"""
LINMA1731's project - Part 1 - Group 18
Authors:
    -Ari Prezerowitz
    -Matya Aydin
"""



###Exercise 1.e###

#We define the sample sizes and the number of repetitions of the experiment
sample_size = [50, 100, 200, 500, 1000, 5000, 10000]
M= 500

#Parameters of the log-normal distribution
mu = 1
sigma = 1


###Asymptotic approach###
mu_MLE_realisations = np.zeros((len(sample_size), M))
sigma_MLE_realisations = np.zeros((len(sample_size), M))

mu_MM_realisations = np.zeros((len(sample_size), M))
sigma_MM_realisations = np.zeros((len(sample_size), M))

for i in range(len(sample_size)):
    N = sample_size[i]
    #We repeat the experiment M times:
    for j in range(M):
        #We generate the N samples:
        data = np.random.lognormal(mu, sigma, N)

        #Estimation by MLE:
        mu_MLE_realisations[i, j] = np.mean(np.log(data))
        sigma_MLE_realisations[i,j] = np.mean(np.square(np.log(data) - mu_MLE_realisations[i,j]))

        #Estimation by MM:
        mu_MM_realisations[i, j] = 2*np.log(np.sum(data)) - 1.5*np.log(N) - 0.5*np.log(np.sum(np.square(data)))
        sigma_MM_realisations[i, j] = np.log(np.sum(np.square(data))) + np.log(N) - 2*np.log(np.sum(data))




#mu_MLE plot:
mean_mu_mle = np.mean(mu_MLE_realisations, axis = 1)
std_mu_mle = np.std(mu_MLE_realisations, axis = 1)
plt.plot(sample_size, mean_mu_mle, label = 'Mean')
plt.fill_between(sample_size, mean_mu_mle - std_mu_mle, mean_mu_mle + std_mu_mle, alpha = 0.3, label = 'Standard deviation')
plt.hlines(y=mu, linewidth=0.5, xmin=sample_size[0], xmax=sample_size[-1], color='black', label="$\mu = 1$", linestyle="--")
plt.title(r'$ \hat{\mu}_{MLE}$' +  f' realisations for {M} experiments')
plt.xlabel('Sample size (N) [-]')
plt.ylabel('Mean of $\mu_{MLE}$ [-]')
plt.ylim(max(mean_mu_mle + std_mu_mle), min( mean_mu_mle - std_mu_mle))
plt.legend(loc='upper right')
#plt.savefig('mu_MLE.pdf')
plt.show()


#sigma_MLE plot:
mean_sigma_mle = np.mean(sigma_MLE_realisations, axis = 1)
std_sigma_mle = np.std(sigma_MLE_realisations, axis = 1)
plt.plot(sample_size, mean_sigma_mle, label = 'Mean')
plt.fill_between(sample_size, mean_sigma_mle - std_sigma_mle, mean_sigma_mle + std_sigma_mle, alpha = 0.3, label = 'Standard deviation')
plt.hlines(y=sigma, linewidth=0.5, xmin=sample_size[0], xmax=sample_size[-1], color='black', label="$\sigma = 1$", linestyle="--")
plt.title(r'$ \hat{\sigma}_{MLE}$' +  f' realisations for {M} experiments')
plt.xlabel('Sample size (N) [-]')
plt.ylabel('Mean of $\sigma_{MLE}$ [-]')
plt.ylim(max(mean_sigma_mle + std_sigma_mle), min( mean_sigma_mle - std_sigma_mle))
plt.legend(loc='upper right')
#plt.savefig('sigma_MLE.pdf')
plt.show()


# mu_MM plot:
mean_mu_mm = np.mean(mu_MM_realisations, axis=1)
std_mu_mm = np.std(mu_MM_realisations, axis=1)
plt.plot(sample_size, mean_mu_mm, label='Mean')
plt.fill_between(sample_size, mean_mu_mm - std_mu_mm, mean_mu_mm + std_mu_mm, alpha=0.3, label='Standard deviation')
plt.hlines(y=mu, linewidth=0.5, xmin=sample_size[0], xmax=sample_size[-1], color='black', label="$\mu = 1$", linestyle="--")
plt.title(r'$ \hat{\mu}_{MM}$' + f' realisations for {M} experiments')
plt.xlabel('Sample size (N) [-]')
plt.ylabel('Mean of $\mu_{MM}$ [-]')
plt.ylim(max(mean_mu_mm + std_mu_mm), min(mean_mu_mm - std_mu_mm))
plt.legend(loc='upper right')
#plt.savefig('mu_MM.pdf')
plt.show()

# sigma_MM plot:
mean_sigma_mm = np.mean(sigma_MM_realisations, axis=1)
std_sigma_mm = np.std(sigma_MM_realisations, axis=1)
plt.plot(sample_size, mean_sigma_mm, label='Mean')
plt.fill_between(sample_size, mean_sigma_mm - std_sigma_mm, mean_sigma_mm + std_sigma_mm, alpha=0.3, label='Standard deviation')
plt.hlines(y=sigma, linewidth=0.5, xmin=sample_size[0], xmax=sample_size[-1], color='black', label="$\sigma = 1$", linestyle="--")
plt.title(r'$ \hat{\sigma}_{MM}$' + f' realisations for {M} experiments')
plt.xlabel('Sample size (N) [-]')
plt.ylabel('Mean of $\sigma_{MM}$ [-]')
plt.ylim(max(mean_sigma_mm + std_sigma_mm), min(mean_sigma_mm - std_sigma_mm))
plt.legend(loc='upper right')
#plt.savefig('sigma_MM.pdf')
plt.show()








###Exercise 1.g###

def matrix_ratio(A, B):
    n = len(A)
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i] = A[i, i] / B[i, i]
    return C

sample_size = np.array([10, 50, 150, 500, 1000, 3000, 5000, 10000])
mu_MLE_realisations = np.zeros(M)
sigma_MLE_realisations = np.zeros(M)

err = np.zeros(len(sample_size))
err_diff = np.zeros(len(sample_size))


for j in range(len(sample_size)):
    N = sample_size[j]
    #M realisations of the MLE on N samples
    inverse_fisher = np.array([
            [sigma**2/N, 0],
            [0, sigma**2/(N*2)]
        ])
    for i in range(M):

        #We generate the N samples:
        data = np.random.lognormal(mu, sigma, N)

        mu_MLE_realisations[i] = np.mean(np.log(data))
        sigma_MLE_realisations[i] = np.sqrt(np.mean(np.square(np.log(data) - mu_MLE_realisations[i])))
    empirical_covariance = np.cov(mu_MLE_realisations, sigma_MLE_realisations)

    ratio = matrix_ratio(empirical_covariance, inverse_fisher)
    sns.heatmap(ratio, annot=True, center=0.5)
    plt.title("Ratio between " +  r'$Cov(\hat{\theta}_{MLE})$ and $I(\theta)$' + f' for N = {N}')
    #plt.savefig(f'heatmap_{N}.pdf')
    plt.show()
    print("Sample size: ", N)
    print(ratio, "\n")
    err_diff[j] = np.linalg.norm(empirical_covariance - inverse_fisher, ord='fro')


plt.xscale('log')
plt.yscale('log')
lin = [1/i for i in sample_size]
plt.plot(sample_size,  lin,linestyle="--", c="black", label="$\mathcal{O}(N^{-1})$")
plt.plot(sample_size, err_diff, label=r'$||Cov(\hat{\theta}_{MLE}) - I(\theta)||_{F}$')
plt.xlabel("Sample size N [-]")
plt.ylabel(" error [-]")
plt.legend(loc='upper right')
plt.grid(True, which="both", ls="--")
plt.title("Error between " + r'$Cov(\hat{\theta}_{MLE})$' + ' and '+ r'$I(\theta)$' + ' for $10^4$ experiments')
#plt.savefig('error_fisher.pdf')
plt.show()





###TCL-like appoach that was finally not used for 1.e###
#Estimated parameters with MLE:
mu_MLE = np.zeros(M)
sigma_MLE = np.zeros(M)

#Estimated parameters with MM:
mu_MM = np.zeros(M)
sigma_MM = np.zeros(M)


#We try different sample sizes:
for N in sample_size:
    #We repeat the experiment M times:
    for i in range(M):
        #We generate the N samples:
        data = np.random.lognormal(mu, sigma, N)

        #Estimation by MLE:
        mu_MLE[i] = np.mean(np.log(data))
        sigma_MLE[i] = np.mean(np.square(np.log(data) - mu_MLE[i]))


        #Estimation by MM:
        mu_MM[i] = 2*np.log(np.sum(data)) - 1.5*np.log(N) - 0.5*np.log(np.sum(np.square(data)))
        sigma_MM[i] = np.log(np.sum(np.square(data))) + np.log(N) - 2*np.log(np.sum(data))

    #mu_MLE plot:
    #plot TCL
    sns.histplot(mu_MLE, kde=True, color='red', label='MLE')
    plt.title(r'$ \hat{\mu}_{MLE}$' +  f' distribution for N = {N} and M = {M}')
    #plt.savefig(f'plot_MLE/mu_MLE_N_{N}_M_{M}.pdf')
    plt.show()




    #sigma_MLE plot:
    #plot TCL
    sns.histplot(sigma_MLE, kde=True, color='red', label='MLE')
    plt.title(r'$ \hat{\sigma}^{2}_{MLE}$' +  f' distribution for N = {N} and M = {M}')
    #plt.savefig(f'plot_MLE/sigma_MLE_N_{N}_M_{M}.pdf')
    plt.show()

    

    

    #mu_MM plot:
    #plot TCL
    sns.histplot(mu_MM, kde=True, color='blue', label='MM')
    plt.title(r'$ \hat{\mu}_{MM}$' +  f' distribution for N = {N} and M = {M}')
    #plt.savefig(f'plot_MM/mu_MM_N_{N}_M_{M}.pdf')
    plt.show()
    #sigma_MM plot:
    #plot TCL
    sns.histplot(sigma_MM, kde=True, color='blue', label='MM')
    plt.title(r'$ \hat{\sigma}^{2}_{MM}$' +  f' distribution for N = {N} and M = {M}')
    #plt.savefig(f'plot_MM/sigma_MM_N_{N}_M_{M}.pdf')
    plt.show()








    

