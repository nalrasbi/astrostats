import numpy as np
# Question 7.1 by using Case II
np.random.seed(623)
x = np.random.normal(130.0, 1.5, size=9)
print(np.mean(x)) # 131.08
n = len(x)
xbar = np.mean(x) # Sample Mean
mu0 = 130.0 # Null value
sample_variance = np.sum((x-xbar)**2)/(n-1)
S = np.sqrt(sample_variance)
print(S)
Z = (xbar - mu0)/(S/np.sqrt(n))
print(Z)
# print(np.std(x,ddof=1))
