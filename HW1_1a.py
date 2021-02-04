# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:05:34 2021

@author: Nasser AL-rasbi
"""
import matplotlib.pyplot as plt
import numpy as np
mu, sigma = 0, 1  # mean and standard deviation
sum = np.zeros([100, ])
for i in range(10000):
    s = np.random.normal(mu, sigma, 100)
    if (i == 1):
        count, bins, ignored = plt.hist(s, 30, density=True)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                 linewidth=2, color='r')
        plt.title("First Graph for the First Normal Dis")
        plt.show()

    sum = np.add(s, sum)
xmean_norm = sum / 10000
print(xmean_norm.shape)
print(xmean_norm)

count, bins, ignored = plt.hist(xmean_norm, 30, density=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
         np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
         linewidth=2, color='r')
plt.title("Mean Normal Dis for 10,000 Number Plot")
plt.show()