import numpy as np
import matplotlib.pyplot as plt
import math
M = 10000
N = 100
p = 0.4

# One experiment
num_success = 0
randx = np.random.random_sample(N)
for x in randx:
    if x < 0.4:
        num_success += 1

num_success

#Simulate 10,000 experiments
ensemble_results1 = np.zeros(M)

for i in range(M):
    num_success = 0

    randx = np.random.random_sample(N)
    for x in randx:
        if x < 0.4:
            num_success += 1

    ensemble_results1[i] = num_success
# One where p can change
p2 = p # Starts at p = 0.4
prev_success, current_success = False, False
num_success = 0

randx = np.random.random_sample(N)
for x in randx:
# Test for success
    if x < p2:
        num_success += 1
        current_success = True
    else:
        current_success = False

# Update probability
    if prev_success and current_success:
        p2 = p*1.1 # 0.44
    else:
        p2 = p # 0.4

    # Update the prev success with current one
    prev_success = current_success

num_success
# 10,000 experiments where p can change
ensemble_results2 = np.zeros(M)

for i in range(M):
    p2 = p # Starts at p = 0.4
    prev_success, current_success = False, False
    num_success = 0

    randx = np.random.random_sample(N)
    for x in randx:# Test for success
        if x < p2:
            num_success += 1
            current_success = True
        else:
            current_success = False
# Update probability
        if prev_success and current_success:
            p2 = p*1.1 # 0.44
        else:
            p2 = p # 0.4

        # Update the prev success with current one
        prev_success = current_success

    ensemble_results2[i] = num_success
#Plot P(k) from Binomial Distribution
def binomial_d(k, N, p):
    combos = math.factorial(N)/(math.factorial(k) * math.factorial(N-k))
    return combos * p**k * (1.0 - p)**(N-k)
klist = np.arange(min(ensemble_results1), max(ensemble_results1)+1, 1)
Pklist = []
for k in klist:
    Pklist.append(binomial_d(k, N, p))
#Plot P(k)
print(min(ensemble_results1), max(ensemble_results1))
bins = np.arange(min(ensemble_results1)-0.5, max(ensemble_results1)+1.5, 1)
plt.figure()
x = plt.hist(ensemble_results1, density=True, bins=bins, label='p is fixed')
# Align = mid so centered on integers
plt.hist(ensemble_results2, density=True, bins=bins, histtype='step', align='mid', label='p can change')
plt.plot(klist, Pklist, label='Binomial D')
plt.xlabel('Number of Successes, $k$');
plt.ylabel('Probability Density, $P(k)$');
plt.title('Probability Density, $P(k)$ vs. Number of Successes, $k$ ');
plt.legend()
plt.savefig('HW4.2.pdf')