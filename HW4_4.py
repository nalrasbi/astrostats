import numpy as np
import matplotlib.pyplot as plt
import math
dataset = np.loadtxt('xray.txt').astype(int)
print(dataset[:10])
# Count the number of times k flares occur in a single day
Nflares_in_day = []
Nflares_today = 1 
Ndays= 1
year, month, day, hour, minute = dataset[0]
for event in dataset[1:]:
    if year == event[0] and month == event[1] and day == event[2]:
        Nflares_today += 1
    else:
        Nflares_in_day.append(Nflares_today)
        Nflares_today = 1
        Ndays += 1
        year, month, day, hour, minute = event
#Compute Poisson PDF
def poisson_pdf(k, lam, t):
    term1 = (lam * t)**k
    term2 = np.exp(-1.0 * lam * t)
    denom = math.factorial(k)
    return term1 * term2 / denom
klist = np.arange(min(Nflares_in_day), max(Nflares_in_day)+1, 1)
Pklist = []
for k in klist:
    Pklist.append(poisson_pdf(k, len(dataset)/(Ndays*24), 24))
bins = np.arange(min(Nflares_in_day)-0.5, max(Nflares_in_day)+1.5, 1)
# Histogram
plt.hist(Nflares_in_day, density=True, bins=bins, color='k', histtype='step', label='X-ray flare data')
plt.xlabel('Number of X-ray flares in a day')
plt.ylabel('Probability Density')
plt.title('The probability of X-ra flare data ocurring and Poisson PDF in a day')
plt.plot(klist, Pklist, '-ok', label='Poisson PDF')
plt.legend()
# Plot the time between flares
import datetime as dt
dataset2 = []
hour0 = dt.datetime(year=2000, month=1, day=1, hour=0)
for event in dataset:
    delta = dt.datetime(year=event[0], month=event[1], day=event[2], hour=event[3]) - hour0
    in_hours = int(delta.total_seconds()/3600)
    dataset2.append(in_hours)
time_between_flares = []
for i in range(1, len(dataset2)):
    delta = dataset2[i] - dataset2[i-1]
    time_between_flares.append(delta)  
plt.figure()
plt.hist(time_between_flares, bins=30, density=True)
plt.xlabel('Hours between X-ray flares')
plt.ylabel('Probability Density')
plt.title('The time between flares')
# The data conform the assumptions for the Poisson Distrbution