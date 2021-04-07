import numpy as np
import matplotlib.pyplot as plt
# 6.2 Confidence Intervals when Sampling Dist is Known
mu = 10
sigma = 1
n = 10
N = 10000
data = np.random.normal(loc=mu, scale=sigma, size=n)
xbar1 = np.mean(data)
delta1 = 1.96 * sigma / np.sqrt(n)
print(f"95% CI on sample mean xbar = {xbar1} +/- {delta1}")
xbar2 = np.mean(data)
talpha = 2.262
sample_std2 = np.std(data, ddof=1)
delta2 = talpha * sample_std2 / np.sqrt(n)
print(f"95% CI on sample mean xbar = {xbar2} +/- {delta2}")
#Check part 1
samples3 = np.random.normal(loc=mu, scale=sigma, size=(N,n))
xbar3 = np.mean(samples3, axis=1)
delta3 = 1.96 * sigma / np.sqrt(n)
percentage3 = np.sum(np.abs(xbar3 - mu) < delta3)/N
print(f"{100*percentage3} CI included mu")
#Check part 2
samples4 = np.random.normal(loc=mu, scale=sigma, size=(N,n))
xbar4 = np.mean(samples4, axis=1)
delta4 = talpha * np.std(samples4, axis=1) / np.sqrt(n)
percentage4 = np.sum(np.abs(xbar4 - mu) < delta4)/N
print(f"{100*percentage4} CI included mu")
# Plot them
plt.figure()
plt.hist(xbar3, bins=100, density=True);
plt.xlabel('$\overline{X}$')
plt.ylabel("P(z)")
plt.title('$\overline{X} = $%.2f +/- %.2f' % (np.mean(xbar3), delta3))
plt.savefig('HW6_2_1.pdf')
plt.figure()
delta5 = 2.262*np.std(samples4)/np.sqrt(n)
plt.hist(xbar4, bins=100, density=True);
plt.xlabel('$\overline{X}$')
plt.ylabel("P(t)")
plt.title('$\overline{X} = $ %.2f +/- %.2f' % (np.mean(xbar4), delta5))
plt.savefig('HW6_2_2.pdf')