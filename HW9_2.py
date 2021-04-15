import numpy as np
import scipy.stats
# The data x and y
x = np.array([-1.22, -1.17, 0.93, -0.58, -1.14])
y = np.array([1.03, -1.59, -0.41, 0.71, 2.10])
# Compute the difference in the means
# print(np.mean(y)-np.mean(x))
print("Difference in the means =", np.mean(y)-np.mean(x))
# Compute the sample standard
sigma_x = np.std(x,ddof=1)
sigma_y = np.std(y,ddof=1)
# Make variable to simplify equations
sigma_xy_pooled = np.sqrt(sigma_x**2 + sigma_y**2)/np.sqrt(2)
# print(sigma_xy_pooled)
# Compute the test statistics for testing the difference in pop means is zero
# Two-tailed test, delta0 = 0
# Devore 358
t = (np.mean(y)-np.mean(x))/(sigma_xy_pooled*np.sqrt(2/5))
# print(t)
print("t statistic =", t)
# Compute the number of degrees of freedom
# Devore 357
a = sigma_x**2/5
b = sigma_y**2/5
nu = 4*(a + b)**2/(a**2 + b**2)
# print(np.floor(nu))
print("dof =", np.floor(nu))
# Compute area under the curve from -inf to -t
f = scipy.stats.t.cdf(-t, nu)
# print(f)
print("cdf =", f)
# Use scipy to perform a t-test on two independent variables
t, p = scipy.stats.ttest_ind(x, y, equal_var=False)
print("t-statistic from scipy =", t)
print("p value from scipy =", p)
# The delta for the confidence interval for difference in the means
# Devore 385
print(t*np.sqrt(a + b))
### Part 2
t_025_6 = 2.447
t_11_6 = 1.333
delta_mu = np.mean(y)-np.mean(x)
delta95 = t_025_6 * np.sqrt(a+b)
delta78 = t_11_6 * np.sqrt(a+b)
print("The 95 percent CI on the difference in the means is [%.2f, %.2f]" % (delta_mu - delta95, delta_mu + delta95))
print("The 78 percent CI on the difference in the means is [%.2f, %.2f]" % (delta_mu - delta78, delta_mu + delta78))