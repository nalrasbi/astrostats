# HW3_3
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 1, 101) # Make a list of 101 numbers between 0 and 1
print(x)
# The first part
def posterior1(theta):
    return 2.0*theta
y1 = posterior1(x)
print(y1)
plt.figure()
plt.plot(x,y1)
plt.xlabel(r'$\theta$');
plt.ylabel(r'$P(\theta|\mathcal{D})$');
plt.title(r'Posterior 1 when $\mathcal{D} = [H]$ and constant prior');
# The second part
def posterior2(theta):
    return 6.0*theta*(1.0 - theta)
y2 = posterior2(x)
print(y2)
plt.figure()
plt.plot(x,y2)
plt.xlabel(r'$\theta$');
plt.ylabel(r'$P(\theta|\mathcal{D})$');
plt.title(r'Posterior 2 when $\mathcal{D} = [H,T]$ and constant prior');
# The third part
def posterior3(theta):
    likelihood = 2.0 * theta * (1.0 - theta)
    prior = np.exp(-(theta - 0.5)**2 / 0.1)
    norm = 1.0
    return likelihood * prior / norm
y3 = posterior3(x)
print(y3)
plt.figure()
plt.plot(x,y3)
plt.xlabel(r'$\theta$');
plt.ylabel(r'$P(\theta|\mathcal{D})$');
plt.title(r'Posterior 3 when $\mathcal{D} = [H,T]$ and Gaussian prior');