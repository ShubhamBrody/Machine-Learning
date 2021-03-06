
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
print(dataset)

N = 10000
d = 10
ads_selected = []
number_of_selections = [0] * d
sums_of_reward = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_UCB = 0
    for i in range(0, d):
        if (number_of_selections[i] > 0):
            average_reward = sums_of_reward[i] / number_of_selections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound > max_UCB):
            max_UCB = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] += 1
    Reward = dataset.values[n, ad]
    sums_of_reward[ad] += Reward
    total_reward += Reward
    
plt.hist(ads_selected)
plt.title("Histogram of Ads Selection")
plt.xlabel("Ads")
plt.ylabel("Number of times each ads were selected")
plt.show()


