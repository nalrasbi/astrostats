import numpy as np
import matplotlib.pyplot as plt
def likelihood(theta):
    x = 0.5
    return np.exp(-0.5 * (x-theta)**2)/np.sqrt(2*np.pi)
def prior(theta):
    if np.abs(theta) > 1:
        return 0
    return 0.5
theta = np.linspace(-1.5, 1.5, 301)
posterior = []
for t in theta:
    posterior.append(prior(t)*likelihood(t))
posterior = np.array(posterior)
# Normalization
normalization = 1.0/np.sum(posterior*(theta[1]-theta[0]))
plt.figure()
plt.plot(theta, posterior*normalization, '-k')
plt.xlabel(r'$\theta$')
plt.ylabel('Probability Density')
plt.title(r'The posterior pdf $ p(\theta|\mathcal{D})$')