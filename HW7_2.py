import numpy as np
import math
# We will use data from example 8.6
n = 9 
mu0 = 130.0 # null value
muprime = 131.5 # Actual value we want to test
sigma = 1.5 # The population standard deviation
z005 = 2.58 # Test statistic value from table A.3
#### The theory
# We will write a formula for the standard
# normal cdf where mean is zero and sigma is 1
def normal_cdf(x):
    return 0.5 * (1.0 + math.erf((x-0.0)/(1.0*np.sqrt(2))))
# We will apply it for alternative hypothesis from P.314
term = (mu0 - muprime)/(sigma/np.sqrt(n))
beta_theory = normal_cdf(z005 + term) - normal_cdf(-z005 + term)
print(beta_theory)
# we will accept null value by this percent even though it's wrong
#### The simulation
# Experiment for 1,000,000 times for example 8.6 to check our theory
N = 1000000
x = np.random.normal(muprime, sigma, size=(N,n))
xbar = np.mean(x, axis=1)
# print(xbar.shape)
# We compute the test statistic
z = (xbar - mu0)/(sigma/np.sqrt(n))
#We check how often z fall between -z005 and z005
counter = 0
for onez in z:
    if -z005 < onez < z005:
        counter += 1
print(counter/N)