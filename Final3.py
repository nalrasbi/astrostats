import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import emcee
from scipy.integrate import dblquad

# Note: My code needs about 2 min to run and I uploaded Final3_data.txt
x, y = np.loadtxt("Final3_data.txt", skiprows=1, unpack=True)

# 12.3.1 Perform a linear regression

tvalue = 2.101   # alpha=0.025 and V=18 (degree of freedom)
linreg = scipy.stats.linregress(x, y)
print("Linear Regression by using Scipy")
print("****************")
print(f"slope = {linreg.slope:.3f}")
print(f"95% CI on slope is [{linreg.slope-tvalue*linreg.stderr:.3f}, {linreg.slope+tvalue*linreg.stderr:.3f}]")
print(f"intercept = {linreg.intercept:.3f}")
print(f"95% CI on intercept is [{linreg.intercept-tvalue*linreg.intercept_stderr:.3f}, {linreg.intercept+tvalue*linreg.intercept_stderr:.3f}]")

def linreg_slope(x, y, axis=None):
    n = x.shape[-1]
    numerator = n * np.sum(x*y, axis=axis) - np.sum(x, axis=axis) * np.sum(y, axis=axis)
    denominator = n * np.sum(x**2, axis=axis) - np.sum(x, axis=axis)**2
    return numerator/denominator

def linreg_intercept(x, y, axis=None):
    n = x.shape[-1]
    slope = linreg_slope(x, y, axis)
    return (np.sum(y, axis=axis) - slope * np.sum(x, axis=axis))/n
a = linreg_intercept(x, y)
b = linreg_slope(x, y)

##http://home.iitk.ac.in/~shalab/regression/Chapter2-Regression-SimpleLinearRegressionAnalysis.pdf
## Pages: 17 and 19

SSres = np.sum((y - (a  + b*x))**2)
sxx = np.sum((x - np.mean(x))**2)
n = len(x)

adeltaa = tvalue * np.sqrt((SSres/(n-2.)) * (1./n + np.mean(x)**2/sxx))
bdeltaa = tvalue * np.sqrt(SSres/((n-2.)*sxx))

print("Linear Regression Analytically  ")
print("****************")
print(f"slope = {b:.3f}")
print(f"95% CI on slope is [{b-bdeltaa:.3f}, {b+bdeltaa:.3f}]")
print(f"intercept = {a:.3f}")
print(f"95% CI on intercept is [{a-adeltaa:.3f}, {a+adeltaa:.3f}]")


# 12.3.2 Analytic Posterior 

def calc_prior(a, b):
    if not -5 <= a <= 5:
        return 0
    if not -5 <= b <= 5:
        return 0
    return 0.01

def calc_likelihood(a, b, x, y, standard_dev):
    r = y - a*x - b
    likelihood = np.exp(-(r**2)/(2.*standard_dev**2))/np.sqrt(2*np.pi*standard_dev**2)
    return np.prod(likelihood)

def calc_posterior(a, b, x, y, standard_dev):
    prior = calc_prior(a, b)
    if prior < 0.001:
        return 0
    
    likelihood = calc_likelihood(a, b, x, y, standard_dev)
    return prior * likelihood

def calc_ln_posterior(theta, x, y, standard_dev):
    a, b = theta
    return np.log(calc_posterior(a, b, x, y, standard_dev))

standard_dev = 0.3
alist = np.linspace(1.25, 2.25, 101)
blist = np.linspace(2.5, 3.5, 101)

analytic_posterior = np.zeros((len(alist), len(blist)))
for i in range(len(alist)):
    for j in range(len(blist)):
        analytic_posterior[i, j] = calc_posterior(alist[i], blist[j], x, y, standard_dev)
        
adelta = alist[1]-alist[0]
bdelta = blist[1]-blist[0]
agrid = np.append(alist, max(alist)+adelta) - adelta/2
bgrid = np.append(blist, max(blist)+bdelta) - bdelta/2

plt.figure()
m = plt.pcolor(bgrid, agrid, analytic_posterior)
plt.xlabel(r'$b$, Intercept')
plt.ylabel(r'$a$, Slope')
plt.title(r'The posterior $ p((a,b)|\mathcal{D})$')



# 12.3.3 Use emcee
##def calc_ln_posterior(theta, x, y, standard_dev)

nstep = 1e4
nwalk = 10
ndims = 2

a_initial = np.random.normal(linreg.slope, 0.1, nwalk)
b_initial = np.random.normal(linreg.intercept, 0.1, nwalk)
thetas_initial = np.array([a_initial, b_initial]).transpose()

sampler = emcee.EnsembleSampler(nwalk, ndims, calc_ln_posterior, args=(x, y, standard_dev))
sampler.run_mcmc(thetas_initial, nstep)
samples = sampler.flatchain


# Find the 95% Credible Interval by using the posterior computed in 2

# scipy.integrate.dblquad
norm = dblquad(calc_likelihood, 2, 4, lambda x: 1., lambda x: 3,
                args=(x, y, standard_dev))[0]

def calc_likelihood_normed(a, b, x, y, standard_dev, norm):
    return calc_likelihood(a, b, x, y, standard_dev)/norm

alist = np.arange(1.3, 2.15, 0.01)
my_list_a = []
for i in range(len(alist)):
    for j in range(len(alist)):
        if alist[i] > 1.5:
            continue
        if alist[j] < 2.0:
            continue
            
        # Do the integral
        prob = dblquad(calc_likelihood_normed, 2, 4, lambda x: alist[i],
                        lambda x: alist[j], args=(x, y, standard_dev, norm))[0]
        
        if np.abs(prob-0.95) < 0.001:
            my_list_a.append([alist[i], alist[j], alist[j]-alist[i], prob])
            
min_idx_a = np.argmin(np.array(my_list_a)[:,2])
a_95_min = my_list_a[min_idx_a][0]
a_95_max = my_list_a[min_idx_a][1]

blist = np.arange(2.8, 3.3, 0.01)
my_list_b = []
for i in range(len(blist)):
    for j in range(len(blist)):
        if blist[i] > 2.975:
            continue
        if blist[j] < 3.15:
            continue
            
        # Do the integral
        prob = dblquad(calc_likelihood_normed, blist[i], blist[j], lambda x:1 ,
                        lambda x:3 , args=(x, y, standard_dev, norm))[0]
        
        if np.abs(prob-0.95) < 0.001:
            my_list_b.append([blist[i], blist[j], blist[j]-blist[i], prob])
            
min_idx_b = np.argmin(np.array(my_list_b)[:,2])
b_95_min = my_list_b[min_idx_b][0]
b_95_max = my_list_b[min_idx_b][1]

# Find the 95% Credible Interval by using the posteriors computed in 3


# asamples = samples[:,0]
# Nsamples = len(asamples)
# alist = np.arange(1.2, 2.4, 0.01)

# my_list_a = []
# for i in range(len(alist)):
#     for j in range(len(alist)):
#         if alist[i] > 1.6:
#             continue
#         if alist[j] < 1.9:
#             continue
            
#         count = 0
#         for a in asamples:
#             if alist[i] <= a <= alist[j]:
#                 count += 1
#         perc = count/Nsamples
        
#         if np.abs(perc-0.95) < 0.001:
#             my_list_a.append([alist[i], alist[j], alist[j]-alist[i], perc])

# min_idx_a = np.argmin(np.array(my_list_a)[:,2])
# a_95_min = my_list_a[min_idx_a][0]
# a_95_max = my_list_a[min_idx_a][1]

# bsamples = samples[:,1]
# Nsamples = len(bsamples)
# blist = np.arange(2.8, 3.3, 0.01)

# my_list_b = []
# for i in range(len(blist)):
#     for j in range(len(blist)):
#         if blist[i] > 2.975:
#             continue
#         if blist[j] < 3.15:
#             continue
            
#         count = 0
#         for b in bsamples:
#             if blist[i] <= b <= blist[j]:
#                 count += 1
#         perc = count/Nsamples
        
#         if np.abs(perc-0.95) < 0.001:
#             my_list_b.append([blist[i], blist[j], blist[j]-blist[i], perc])

# min_idx_b = np.argmin(np.array(my_list_b)[:,2])
# b_95_min = my_list_b[min_idx_b][0]
# b_95_max = my_list_b[min_idx_b][1]

# Plot

plt.figure()
plt.hist(samples[:,0], density=True, bins=50)
plt.axvline(a_95_min, color='k')
plt.axvline(a_95_max, color='k')
plt.xlabel("a, slope")
plt.ylabel("Probability Density")
plt.title(f"Posterior for a, 95% CI is [{a_95_min:.2f}, {a_95_max:.2f}]")

plt.figure()
plt.hist(samples[:,1], density=True, bins=50)
plt.axvline(b_95_min, color='k')
plt.axvline(b_95_max, color='k')
plt.xlabel("b, intercept")
plt.ylabel("Probability Density")
plt.title(f"Posterior for b, 95% CI is [{b_95_min:.2f}, {b_95_max:.2f}]")