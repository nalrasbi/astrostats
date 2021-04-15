import numpy as np
import matplotlib.pyplot as plt
def calc_prior(theta):
    if np.abs(theta) > 1:
        return 0
    return 0.5
def calc_normal(x, mean, std):
    return np.exp(-(x-mean)**2/(2*std**2))/np.sqrt(2*np.pi*std**2)
def calc_likelihood(data, theta):
    sol = calc_normal(data, theta, 1)
    return np.prod(sol)
data = np.array([0.5, 1.5])
thetalist = np.linspace(-1.01, 1.01, 203)
delta = thetalist[1]-thetalist[0]
# Analytic Posterior
posterior_exact = []
for theta in thetalist:
    post = calc_prior(theta) * calc_likelihood(data, theta)
    posterior_exact.append(post)
posterior_exact = np.array(posterior_exact)
# Simulation Post = prob(near r0)*prob(near r1)
N = 1000000
posterior_sim = []
for theta in thetalist:
    # check prior
    prior = calc_prior(theta)
    if prior < 0.25:
        posterior_sim.append(0)
        continue
   # Prob(random num near r0)
    r0 = np.random.normal(theta, 1, size=N)
    p0 = np.sum(np.abs(r0 - data[0]) <= delta/2)/N
   # Prob(random num near r1)
    r1 = np.random.normal(theta, 1, size=N)
    p1 = np.sum(np.abs(r1 - data[1]) <= delta/2)/N
    post = prior * p0 * p1
    posterior_sim.append(post)
posterior_sim = np.array(posterior_sim)
norm_exact = np.sum(posterior_exact)*delta
norm_sim = np.sum(posterior_sim)*delta
plt.figure()
plt.plot(thetalist, posterior_exact/norm_exact, '-k', label='Analytic')
plt.bar(thetalist, posterior_sim/norm_sim, width=delta, label='Simulation')
plt.xlabel(r'$\theta$, mean')
plt.ylabel('Probability Density')
plt.title(r'The posterior pdf $ p(\theta|\mathcal{D})$')
plt.legend()