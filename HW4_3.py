import numpy as np
import matplotlib.pyplot as plt
import math
# 4.3 The Poisson Distribution
prob_flare_per_hour = 900. / (1000. * 24.)
Ndays = 1000
dataset= []
for i in range(Ndays*24):
    x = np.random.random_sample()
    if x < prob_flare_per_hour:
        dataset.append(i)      
print(f"In {Ndays} days, we had {len(dataset)} flares")
Nflares = len(dataset)
Nflares_in_day = []
Nflares_today = 0
hours_today = 0 
for i in range(Ndays*24):
    hours_today += 1
    if i in dataset:
        Nflares_today += 1  
    if hours_today == 24:
        Nflares_in_day.append(Nflares_today)
        hours_today = 0
        Nflares_today = 0
def poisson_pdf(k, lam, t):
    term1 = (lam * t)**k
    term2 = np.exp(-1.0 * lam * t)
    denom = math.factorial(k)
    return term1 * term2 / denom
klist = np.arange(min(Nflares_in_day), max(Nflares_in_day)+1, 1)
Pklist = []
for k in klist:
    Pklist.append(poisson_pdf(k, prob_flare_per_hour, 24))
bins = np.arange(-0.5, 6.5, 1)
plt.figure()
plt.hist(Nflares_in_day, density=True, bins=bins, label='Data')
plt.plot(klist, Pklist, '-ok', label='Poisson PDF')
plt.xlabel('Number of X-ray flares in a day')
plt.ylabel('Probability Density')
plt.title('The probability of k flare events ocurring and Poisson PDF per a day')
plt.legend()
# Time between flares
time_between_flares = []
for i in range(1, len(dataset)):
    delta = dataset[i] - dataset[i-1]
    time_between_flares.append(delta)
plt.figure()
plt.hist(time_between_flares, bins=30)
plt.xlabel('Hours between X-ray flares')
plt.ylabel('Counts')
plt.title('The time between flares')