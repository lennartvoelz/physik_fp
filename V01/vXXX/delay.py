import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import pandas as pd

delay, counts = np.loadtxt('delay.csv', unpack=True, delimiter=',')

counts_err = np.sqrt(counts)

counts = unp.uarray(counts, counts_err)

hitrate = counts / 100

plt.errorbar(delay, unp.nominal_values(hitrate), yerr=unp.std_devs(hitrate), fmt='o', color='red', ecolor='black', linestyle='', linewidth=1.5, markersize=4, label='Messwerte mit Poissonfehler')

plt.xlabel('Delay')
plt.ylabel('Hitrate')
plt.title('Hitrate vs. Delay')
plt.grid(True)
plt.gca().set_facecolor('#f7f7f7')

#select all events that have max difference of 0.8 to neighbouring events

treshold = 0.8

def select_events(delay, hitrate, treshold):
    selected_delay = []
    selected_hitrate = []
    for i in range(1, len(delay)-1):
        if (abs(hitrate[i] - hitrate[i-1]) + abs(hitrate[i] - hitrate[i+1])) < treshold:
            selected_delay.append(delay[i])
            selected_hitrate.append(hitrate[i])
    return selected_delay, selected_hitrate

selected_delay, selected_hitrate = select_events(delay, hitrate, treshold)

selected_hitrate = np.array(selected_hitrate)
selected_delay = np.array(selected_delay)

#mean = np.mean(selected_hitrate)
#std = np.std(selected_hitrate)
#
##plot area around mean
#plt.fill_between(selected_delay, mean-std, mean+std, color='blue', alpha=0.3, label='1$\sigma$-Umgebung')

print(selected_delay)
print(selected_hitrate)

plt.legend()

#plt.savefig('delay.pdf')