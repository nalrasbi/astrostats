import numpy as np
import matplotlib.pyplot as plt
import emcee
import scipy.stats as stats

# Note: My code needs about 7 min to run
# Noted ;)

D = [4.72, 9.11, 5.57, 7.66, 4.69, 4.86, 6.51, 6.65, 4.77, 0.26, 4.67, 6.37, 5.05, 5.91,
5.54, 2.13, 6.77, 3.84, 4., 6.18, 3.54, 5.52, 3.29, 4.62, 4.25, 4.08, 3.37, 4.91,
5.24, 6.85, 3.85, 5.11, 9.41, 5.78, 5.97, 5.87, 1.59, 4.51, 0.72, 6.72, 8.4, 3.94,
8.53, 2.76, 2.62, 6.11, 3.37, 4.01, 7.17, 3.05]


# 12.2.1 Frequentist
# Devore, A-11, Table A.7, for v > 40
def chisq_crit(dof, end):
    # Could use
    # scipy.stats.chi2.ppf(0.975, n-1) 
    # scipy.stats.chi2.ppf(0.025, n-1) 
    if end == "lower":
        zalpha = 1.96
    elif end == "upper":
        zalpha = -1.96
    return dof*(1. - 2./(9.*dof) + zalpha*np.sqrt(2./(9.*dof)))**3

# Enough data to say sample var is close to pop variance
# Could use t = scipy.stats.t.ppf(0.975, n-1) as done in HW 9.2 for exact.
# Difference is about 0.02, so assumption OK.
N = len(D)
sample_mean = np.mean(D)
sample_std = np.std(D, ddof=1)
# print(N)
zalpha = 1.96 

delta_mean = zalpha * sample_std/np.sqrt(N)     # Page 277 Devore
print(f"95% CI on the mean, xbar in [{sample_mean-delta_mean:.2f}, {sample_mean+delta_mean:.2f}]")

# Page 295 Devore
# Lower limit 
var_freq_lower = (N-1) * sample_std**2/chisq_crit(49, "lower")
# Upper limit 
var_freq_upper = (N-1) * sample_std**2/chisq_crit(49, "upper")

print(f"95% CI on the std, sigma in [{np.sqrt(var_freq_lower):.2f}, {np.sqrt(var_freq_upper):.2f}]")

#12.2.2  Bayesian

def calc_ln_prior(mu, sigma, priors):
    if not priors["mu_low"] <= mu <= priors["mu_high"]:
        return -np.inf
    if not 0 < sigma <= priors["sigma"]:
        return -np.inf
    return 0

def calc_ln_likelihood(mu, sigma, D):
    lnsigma = np.log(sigma)
    lnlike = -lnsigma - (D-mu)**2/(2*sigma**2)# Simplify it mathematically by hand
    return np.sum(lnlike)

def calc_ln_posterior(theta, D, priors):
    mu, sigma = theta
    
    lnprior = calc_ln_prior(mu, sigma, priors)
    if np.isinf(lnprior):
        return -np.inf
    
    lnlike = calc_ln_likelihood(mu, sigma, D)
    return lnprior + lnlike

nstep = 1e4
nwalk = 10
ndims = 2
priors = {
    "mu_low": 3,
    "mu_high": 8,
    "sigma": 5
}
mu_initial = np.random.uniform(priors["mu_low"], priors["mu_high"], nwalk)
sigma_initial = np.random.uniform(0, priors["sigma"], nwalk)
thetas_initial = np.array([mu_initial, sigma_initial]).transpose()
sampler = emcee.EnsembleSampler(nwalk, ndims, calc_ln_posterior, args=(D, priors))
sampler.run_mcmc(thetas_initial, nstep)
samples = sampler.flatchain

# Find the 95% Credible Interval 

musamples = samples[:,0]
Nsamples = len(musamples)
mulist = np.arange(4.25, 5.75, 0.01)

my_list_mu = []
for i in range(len(mulist)):
    for j in range(len(mulist)):
        # Quick check to speed it up
        if mulist[i] > 4.75:
            continue
        if mulist[j] < 5.25:
            continue
            
        count = 0
        for mu in musamples:
            if mulist[i] <= mu <= mulist[j]:
                count += 1
        perc = count/Nsamples
        
        if np.abs(perc-0.95) < 0.001:
            my_list_mu.append([mulist[i], mulist[j], mulist[j]-mulist[i], perc])

min_idx_mu  = np.argmin(np.array(my_list_mu)[:,2])
mu_95_min = my_list_mu[min_idx_mu][0]
mu_95_max = my_list_mu[min_idx_mu][1]


sigmasamples = samples[:,1]
Nsamples = len(sigmasamples)
sigmalist = np.arange(1.25, 2.75, 0.01)

my_list_sigma = []
for i in range(len(sigmalist)):
    for j in range(len(sigmalist)):
        # Quick check to speed it up
        if sigmalist[i] > 1.75:
            continue
        if sigmalist[j] < 2.25:
            continue
            
        count = 0
        for sigma in sigmasamples:
            if sigmalist[i] <= sigma <= sigmalist[j]:
                count += 1
        perc = count/Nsamples
        
        if np.abs(perc-0.95) < 0.01:
            my_list_sigma.append([sigmalist[i], sigmalist[j], sigmalist[j]-sigmalist[i], perc])
            
min_idx_sigma  = np.argmin(np.array(my_list_sigma)[:,2])
sigma_95_min = my_list_sigma[min_idx_sigma][0]
sigma_95_max = my_list_sigma[min_idx_sigma][1]

# Note:This only to check my solutions
mean, var, std = stats.bayes_mvs(D, 0.95)
# print(mean)
# print(std)

# Figures

plt.figure()
plt.hist(samples[:,0], density=True, bins=50)
plt.title(f'$p(\\mu|D)$ with 95% CI [{mu_95_min:.2f}, {mu_95_max:.2f}]')
plt.axvline(mu_95_min, color='k')
plt.axvline(mu_95_max, color='k')
plt.ylabel('Probability Density')
plt.xlabel(r'$\mu$')

plt.figure()
plt.hist(samples[:,1], density=True, bins=50)
plt.title(f'$p(\\sigma|D)$ with 95% CI [{sigma_95_min:.2f}, {sigma_95_max:.2f}]')
plt.axvline(sigma_95_min, color='k')
plt.axvline(sigma_95_max, color='k')
plt.ylabel('Probability Density')
plt.xlabel(r'$\sigma$')
