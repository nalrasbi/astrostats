import numpy as np
import matplotlib.pyplot as plt

# Compute posterior analytically
theta = np.linspace(0, 1, 101)
posterior_a = 4*3*theta**2*(1-theta)

# Compute posterior via simulation
n_flips, n_heads = 3, 2
N = 100000
posterior_b = []
for t in theta:
    x = np.random.binomial(n_flips, t, size=N)
    n_matches = np.sum(x == n_heads)
    posterior_b.append(n_matches/N)
posterior_b = np.array(posterior_b)

### Part 2
def integral(low, high):
    return 12*((high**3/3 - high**4/4) - (low**3/3 - low**4/4))

lowers = np.linspace(0,1,101)
intervals = np.linspace(0,1,101)
my_list = []
for low_end in lowers:
    for interval in intervals:
        area = integral(low_end, low_end+interval)
        my_list.append([low_end, interval, area])

delta = 0.001
close_my_list = []
for line in my_list:
    if np.abs(line[2]-0.95) < delta:
        close_my_list.append(line)
close_my_list = np.array(close_my_list)
# print(close_my_list)
min_index = np.argmin(close_my_list[:,1])
CI_low = close_my_list[min_index][0]
CI_high = close_my_list[min_index][0] + close_my_list[min_index][1]
# print(CI_low, CI_high)

### Plot
width = theta[1]-theta[0]
normalization = 1/np.sum(posterior_b*0.01)
plt.figure()
plt.plot(theta, posterior_a, '-k', label='Analytic')
plt.bar(theta, posterior_b*normalization, width=width, label='Simulation')
plt.title(r"95 percent Credible Interval for $\theta$ is [%.2f, %.2f]" % (CI_low, CI_high))
plt.axvline(CI_low, color='k', linestyle='--')
plt.axvline(CI_high, color='k', linestyle='--')
plt.xlabel(r'$\theta$')
plt.ylabel('Probability Density')
plt.legend()