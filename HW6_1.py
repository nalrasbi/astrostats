import numpy as np
import matplotlib.pyplot as plt
mu = 1
sigma = 1
n = 10
N = 10000
# Do it once
Y = np.sum(np.random.normal(loc=mu, scale=sigma, size=n)**2)
print(Y)
# Do it For 10,000
random_draws = np.random.normal(loc=mu, scale=sigma, size=(N,n))
print(random_draws.shape)
Y = np.sum(random_draws**2, axis=1)
print(Y.shape)
# Part 2
# values1 = np.array([-0.546, -0.406, -0.115, -1.262, -1.386,
#             -0.448,  0.829,  0.799, -1.100, 0.385])
values1 = np.array([3.03399726, -0.41833714, -0.34603477,  0.84167495,  1.46829221, 
            0.79846543,  1.11989497,  1.55009339,  1.54485525,  1.26485924])
Y_one = np.sum(values1**2)
print(Y_one)
bootstrapped_values = np.random.choice(values1, replace=True, size=(N,n))
Ystar = np.sum(bootstrapped_values**2, axis=1)
print(Ystar.shape)
# Part 3
bins = np.linspace(min([min(Y), min(Ystar)]), max([max(Y), max(Ystar)]), 100)
plt.figure()
plt.hist(Y, density=True, alpha=0.5, bins=bins, label='Y');
plt.hist(Ystar, density=True, alpha=0.5, bins=bins, label=r'$Y^*$')
plt.xlabel(r'Y or $Y^*$')
plt.ylabel("PDF")
plt.title(f'Bootstrapping Trailas for N={N},n={n}')
plt.legend()
plt.savefig('HW6_1_1.pdf')
# Part 3, The scatter plot
pdf1, _ = np.histogram(Y, bins=bins, density=True)
pdf1star, _ = np.histogram(Ystar, bins=bins, density=True)
plt.figure()
plt.plot(pdf1, pdf1star, '.')
plt.xlabel(r'Y or $Y^*$')
plt.ylabel("PDF")
plt.title(f'Bootstrapping Trailas for N={N},n={n}')
plt.savefig('HW6_1_2.pdf')