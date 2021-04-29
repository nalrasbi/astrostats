import numpy as np
from matplotlib import pyplot as plt
import emcee

# HW10_2
N = 10000

# N = 1000000
if False:
    D = [0.0]
    dmu = 0.5
    mu = np.arange(-5, 5 + dmu, dmu)
    P = np.exp( -(mu**2)/2 )/np.sqrt(2*np.pi)
    sample = np.random.normal(0, 1, N)

if True:
    D = [0.5, 1.5]
    dmu = 0.05
    #mu = np.arange(-3, 4 + dmu, dmu)
    mu = np.arange(-1, 1 + dmu, dmu)
    p1 = np.exp( -((mu-D[0])**2)/2 )/np.sqrt(2*np.pi)
    p2 = np.exp( -((mu-D[1])**2)/2 )/np.sqrt(2*np.pi)
    
    P = p1*p2
    P = P/np.sum(P*dmu)
    sample = np.random.normal((D[0] + D[1])/2, 1/np.sqrt(2), N)


i_mu = 1 # Initial index for mu

step_size = 1


# History of steps
i_mu_hist = np.zeros(N, dtype=np.int64)
for i in range(N):

    # "First opinion" fair coin toss
    ht = np.random.binomial(1, 0.5)

    P_right = P[i_mu+1]
    P_curr = P[i_mu]
    P_left  = P[i_mu-1]

    if ht == 1:
        flip1 = "H"
        if P_right > P_curr:
            # If first opinion coin toss says step right and P is higher to right, step right
            flip2 = "N/A" # Not Applicable
            step_dir = 1
        else:
            # If first opinion coin toss says step right and P is lower to right, flip biased
            # coin with p_heads = P_right/P_curr as second opinion.
            ht = np.random.binomial(1, P_right/P_curr)
            if ht == 1:
                flip2 = "H"
                # Second opinion "H" means follow first opinion instructions
                step_dir = 1
            else:
                flip2 = "T"
                # Don't take a step
                step_dir = 0
    else:
        flip1 = "T"
        if P_left > P_curr:
            # If first opinion coin toss says step left and P is higher to left, step left
            flip2 = "N/A"
            step_dir = -1
        else:
            # If first opinion coin toss says step left and P is lower to left, flip biased coin
            # with p_heads = P_left/P_curr for second opinion.
            ht = np.random.binomial(1, P_left/P_curr)
            if ht == 1:
                flip2 = "H" 
                # Second opinion "H" means follow first opinion instructions
                step_dir = -1
            else:
                flip2 = "T"
                # Don't take a step
                step_dir = 0

    # Take step
    if step_dir != 0:
        i_mu = i_mu + step_size*step_dir

    # Handle steps out-of-bounds by not taking a step
    if i_mu < 1:
        i_mu = 0
    if i_mu > mu.shape[0] - 2:
        i_mu = mu.shape[0] - 2

    i_mu_hist[i] = i_mu


# Remove first 10% of history of steps
a = np.int(np.round(0.1*N))
i_mu_hist_r = i_mu_hist[a:]


#HW11_reference
D = [0.5, 1.5]

def log_LxP(theta, D):
    """Return Log( Likelihood * Posterior) given data D."""
    p1 = np.exp( -((theta-D[0])**2)/2 )
    p2 = np.exp( -((theta-D[1])**2)/2 )
    if np.abs(theta) <= 1:
        LxP = p1*p2
    else:
        LxP = 0.0
    return np.log(LxP)

nstep = 1e3     # Number of steps each walker takes
# nstep = 1e4
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


## The Comparison
 
plt.figure()
plt.grid()
bins = 50
bins = mu - dmu/2 # Center bins on possible x positions
plt.hist(allsamples, density=True, bins=30, label='emcee');
plt.hist(mu[i_mu_hist_r], bins=bins, density=True, histtype='step',
         label='Metropolis sampling')
plt.hist(sample, bins=bins, density=True,  histtype='step',
         label='Direct sampling from Exact')
plt.plot(mu, P, 'k',label='Exact ($\mathcal{N}(\mu=1, \sigma^2=1/2)$)') 
plt.ylabel('$p(\\theta|\\mathcal{D})$')
plt.xlabel('$\\theta$')
plt.title('N = {0:d}; $\mathcal{{D}}=[0.5, 1.5]$'.format(N))
plt.legend(loc='upper left')