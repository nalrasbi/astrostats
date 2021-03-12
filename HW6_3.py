import numpy as np
import matplotlib.pyplot as plt
values = [-0.54603241, -0.40652794, -0.11570264, -1.26244673, -1.38616981, -0.44812319,
           0.82880132, 0.79937713, -1.098357, 0.38530288]
N = 10000
n = 10
# Compute some quantities
bootstrap_dist = np.random.choice(values, replace=True, size=(N,n))
sample_means = np.mean(bootstrap_dist, axis=1)
p3mean = np.mean(sample_means)
p3ci = 2.262 * np.std(bootstrap_dist) / np.sqrt(n)
# Plot
plt.figure()
plt.hist(sample_means, density=True, bins=100, label='Sampling Dist')
plt.axvline(p3mean - p3ci, color='k', label='95% CI')
plt.axvline(p3mean + p3ci, color='k')
plt.xlabel('$\overline{X} $')
plt.ylabel("PDF")
plt.title('$\overline{X} = $%.2f +/- %0.2f' % (p3mean, p3ci))
plt.legend()
plt.savefig('HW6_3.pdf')