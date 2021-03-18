import numpy as np
# Question 7.1 by using case III
np.random.seed(623)
x = np.random.normal(130.0, 1.5, size=9)
print(np.mean(x))
n = len(x)
xbar = np.mean(x)
mu0 = 130.
sample_variance = np.sum((x - xbar)**2)/(n-1)
s = np.sqrt(sample_variance)
T = (xbar - mu0)/(s/np.sqrt(n))
t005c8 = 3.355
print(T)