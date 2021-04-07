import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
# Part 1
N = 1000
alpha = 1
beta = 1
mu = 0
sigma = 0.2
B = 10000
n = 20
xi = np.arange(N)/N
yi = beta * xi + alpha + np.random.normal(mu, sigma, size=N)
plt.figure()
plt.plot(xi, yi, '.k')
plt.xlabel('X')
plt.ylabel('Y')
# Most readers will know it is a scatter plot, so it is not needed in title.
plt.title('N (x,y) population values and n = 20 randomly drawn (x,y) pairs')
# Part 2
indicies = np.random.choice(range(N), replace=True, size=n)
plt.plot(xi[indicies], yi[indicies], 'or')
plt.savefig('Midterm_Part 1 and 2.pdf')
# Part 3
def linreg_slope(x, y, axis=None):
    n = x.shape[-1]
    numerator = n * np.sum(x*y, axis=axis) - np.sum(x, axis=axis) * np.sum(y, axis=axis)
    denominator = n * np.sum(x**2, axis=axis) - np.sum(x, axis=axis)**2
    return numerator/denominator
def linreg_intercept(x, y, axis=None):
    n = x.shape[-1]
    slope = linreg_slope(x, y, axis)
    return (np.sum(y, axis=axis) - slope * np.sum(x, axis=axis))/n
print("My b = %.4f" % linreg_slope(xi[indicies], yi[indicies]))
print("My a = %.4f" % linreg_intercept(xi[indicies], yi[indicies]))
# Part 4
scipy_solution = stats.linregress(xi[indicies], yi[indicies])
print("Scipy b = %.4f" % scipy_solution.slope)
print("Scipy a = %.4f" % scipy_solution.intercept)
# Part 5
indicies2d = np.random.choice(range(N), replace=True, size=(B,n))
blist = linreg_slope(xi[indicies2d], yi[indicies2d], axis=1)
alist = linreg_intercept(xi[indicies2d], yi[indicies2d], axis=1)
meana = np.mean(alist)
adelta_estimate = sorted(np.abs(alist - meana))[int(B*0.95)]
meanb = np.mean(blist)
bdelta_estimate = sorted(np.abs(blist - meanb))[int(B*0.95)]
bins = np.linspace(0.4, 1.6, 100)
plt.figure()
plt.hist(alist, bins=bins, density=True, alpha=0.25, color='blue', label='intercept (a)')
plt.axvline(meana-adelta_estimate, color='blue')
plt.axvline(meana+adelta_estimate, color='blue')
plt.hist(blist, bins=bins, density=True, alpha=0.25, color='black', label='slope (b)')
plt.axvline(meanb-bdelta_estimate, color='black')
plt.axvline(meanb+bdelta_estimate, color='black')
plt.legend()
plt.title(r"Estimate 95 Percent CI: $a\in[%.2f, %.2f]$ and $b\in[%.2f, %.2f]$" % (
         meana - adelta_estimate,
         meana + adelta_estimate,
         meanb - bdelta_estimate,
         meanb + bdelta_estimate))
plt.xlabel('Slope or Intercept ')
plt.ylabel('PDF')
plt.savefig('Midterm_Part 5.pdf')
# Part 6
tvalue = 2.101 # alpha=0.025 and V=18 (degree of freedom)
xlist = xi[indicies]
ylist = yi[indicies]
a = linreg_intercept(xlist, ylist)
b = linreg_slope(xlist, ylist)
# Great job. Very few students did this part correctly.
SSres = np.sum((ylist - (a  + b*xlist))**2.0)
sxx = np.sum((xlist - np.mean(xlist))**2.0)
adelta = tvalue * np.sqrt((SSres/(n-2.0)) * (1.0/n + np.mean(xlist)**2.0/sxx))
bdelta = tvalue * np.sqrt(SSres/((n-2.0)*sxx))
print("95 percent CI of a in [%.2f, %.2f]" % (a-adelta, a+adelta))
print("95 percent CI of b in [%.2f, %.2f]" % (b-bdelta, b+bdelta))
if False:
    #AttributeError: 'LinregressResult' object has no attribute 'intercept_stderr'
    print("95 percent CI of a by scipy in [%.2f, %.2f]"\
          %((scipy_solution.intercept-(tvalue*scipy_solution.intercept_stderr)\
             ,scipy_solution.intercept+(tvalue*scipy_solution.intercept_stderr))))
print("95 percent CI of b by scipy in [%.2f, %.2f]"\
      %((scipy_solution.slope-(tvalue*scipy_solution.stderr)\
         ,scipy_solution.slope+(tvalue*scipy_solution.stderr))))

    # Part8
err_on_a = alist - alpha
err_on_b = blist - beta
plt.figure()
plt.plot(err_on_a, err_on_b, '.k')
plt.title(r'$\delta b $ VS $\delta a $')
plt.xlabel(r'$\delta a = a - \alpha$')
plt.ylabel(r'$\delta b = b - \beta$')