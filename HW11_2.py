import numpy as np
import emcee
from matplotlib import pyplot as plt

# D = [0.5, 1.5]
# D = np.random.normal(1, 1, size=10)
# D = np.random.uniform(1, 1.5, size=10)
D = np.random.chisquare(10, size=10)

def log_LxP(theta, D):
    """Return Log( Likelihood * Posterior) given data D."""
#     if theta > 1 or theta < -1:
#         return -np.inf
    
    # Where does this equation come from?
    lnp = -(theta - D)**2/2
    return np.sum(lnp)
    
    
nstep = 1e4     # Number of steps each walker takes
nwalk = 10      # Number of initial values for theta
ndims = 1       # Number of unknown parameters in theta vector

# Create a set of 10 inital values for theta. Values are drawn from
# a distribution that is unform in range [-1, 1] and zero outside.
thetas_initial =  np.random.uniform(-1, 1, (nwalk, ndims))

# Initialize the sampler object
sampler = emcee.EnsembleSampler(nwalk, ndims, log_LxP, args=(D, ))

# Run the MCMC algorithm for each initial theta for 5000 steps
sampler.run_mcmc(thetas_initial, nstep, progress=True);

# Get the values of theta at each step
samples = sampler.get_chain()
allsamples = sampler.flatchain


# My plot

plt.figure()
plt.grid()
plt.hist(allsamples, density=True, bins=30, label='emcee')
plt.ylabel('$p(\\theta|\\mathcal{D})$')
plt.xlabel('$\\theta$')
plt.title('D = np.random.chisquare(10, size=10)')
plt.legend(loc='upper left')