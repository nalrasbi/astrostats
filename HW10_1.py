import numpy as np
import matplotlib.pyplot as plt

def calc_likelihood(data, theta):
    sigma = 1.0
    terms = np.exp(-(data-theta)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    return np.prod(terms)

def calc_prior(theta):
    if -1.0 <= theta <= 1.0:
        return 0.5
    return 0

def calc_post(data, theta):
    return calc_prior(theta) * calc_likelihood(data, theta)

def run_mcmc(data, N, theta_start, prop_width=0.1):
    # Make sure data is a numpy array
    data = np.array(data)
    
    # Initialize some arrays
    chain, posterior = np.empty(N), np.empty(N)
    chain.fill(np.nan)
    posterior.fill(np.nan)
    
    # Find the starting point
    theta_current = theta_start
    post_current = calc_post(data, theta_start)
    
    # Do the MCMC
    for i in range(N):
        accept = False
        while(True):
            # Make a step proposal
            proposal = np.random.normal(theta_current, prop_width)
            post_proposal = calc_post(data, proposal)
            ratio = post_proposal / post_current
            
            # Should we accept the proposal step?
            if ratio > 1.0:
                accept = True
            elif np.random.rand() < ratio:
                accept = True
            
            # Accept the step
            if accept:
                chain[i] = proposal
                posterior[i] = post_proposal
                theta_current = proposal
                post_current = post_proposal
                break
    
    # Return the chain and posterior
    return chain, posterior

# Do you know why there is a mis-match near theta=1? You'll probably be
# able to figure it out by printing out diagnostics as I did in my solution.
# Run the MCMC
N = 100000
theta_start = 1.0
data = [0.5, 1.5]
chain, post = run_mcmc(data, N, theta_start, prop_width=0.1)
# Compute and plot analytic posterior
posterior_exact = []
thetalist_exact = np.linspace(-1.01, 1.01, 202)
for theta_i in thetalist_exact:
    posterior_exact.append(calc_post(data, theta_i))
posterior_exact = np.array(posterior_exact)
norm = np.sum(posterior_exact)*(thetalist_exact[1]-thetalist_exact[0])

# Make the plot
plt.figure()
plt.hist(chain, density=True, bins=100,label='MCMC')
plt.plot(thetalist_exact, posterior_exact/norm, '-k',label='Analytic')
plt.xlabel(r'$\theta$, mean')
plt.ylabel(r'$P(\theta|\mathcal{D})$')
plt.title(f' N={N}; 'r'$\mathcal{D}$='f'{data}')
plt.legend()
# Do the MCMC for 50 random points
N = 100000
theta_start = 0.5
data = np.random.normal(0.5, 1, 50)
chain, post = run_mcmc(data, N, theta_start, prop_width=0.1)

# Compute the analytic posterior for 50 random points
posterior_exact1 = []
thetalist_exact1 = np.linspace(-1.01, 1.01, 203)
for theta_i in thetalist_exact1:
    posterior_exact1.append(calc_post(data, theta_i))
posterior_exact1 = np.array(posterior_exact1)

norm = np.sum(posterior_exact1)*(thetalist_exact1[1]-thetalist_exact1[0])

# Make the plot for 50 random points

plt.figure()
plt.hist(chain, density=True, bins=100,label='MCMC')
plt.plot(thetalist_exact1, posterior_exact1/norm, '-k',label='Analytic')
plt.xlabel(r'$\theta$, mean')
plt.ylabel(r'$P(\theta|\mathcal{D=50})$')
plt.title(f' N={N}; 'r'$\mathcal{D}$= 50 random points')
plt.legend()