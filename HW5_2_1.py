import numpy as np
import matplotlib.pyplot as plt
# 5.2.1
n = 10
M = 10000
def calc_Sbsq(xlist):
    xbar = np.mean(xlist)
    return np.sum((xlist - xbar)**2) / len(xlist)
# Experiment for n=10
xlist = np.random.normal(loc=0, scale=1.0, size=n)
Sbsq = calc_Sbsq(xlist)
print(Sbsq)
# Repeat it for 10,000 times
ensemble1 = np.zeros(M)
for i in range(M):
    xlist = np.random.normal(loc=0, scale=1.0, size=n)
    Sbsq = calc_Sbsq(xlist)
    ensemble1[i] = Sbsq
# Plot a histogram of 10,000 experiments
avg = np.mean(ensemble1)
var = np.var(ensemble1)
plt.figure()
plt.hist(ensemble1, density=True, bins=50)
plt.xlabel(r'$S_b^2$')
plt.ylabel("Probability")
plt.title(f'Average={avg}, Variance={var}')
plt.savefig('HW5_2_1.pdf')