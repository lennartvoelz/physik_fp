import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import pandas as pd

delay, counts = np.loadtxt('../data/delay.csv', unpack=True, delimiter=',')

counts_err = np.sqrt(counts)

counts = unp.uarray(counts, counts_err)

messzeit = unc.ufloat(100, 0.1)

hitrate = counts / messzeit

plt.errorbar(delay, unp.nominal_values(hitrate), yerr=unp.std_devs(hitrate), fmt='o', color='red', ecolor='black', linestyle='', linewidth=1.5, markersize=4, label='Messwerte mit Poissonfehler')

plt.xlabel('Delay [ns]')
plt.ylabel('Hitrate')
plt.grid(True)

selected_hits = hitrate[11:17]
selected_delays = delay[11:17]

mean = np.mean(selected_hits)

plt.fill_between(selected_delays, unp.nominal_values(mean)-unp.std_devs(mean), unp.nominal_values(mean)+unp.std_devs(mean), color='blue', alpha=0.3, label='Mittelwert des Plateaus mit Standardfehler')

print('Mittelwert des Plateaus:', mean)

fwhm_y = mean / 2

def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b*(x))) + c

params, covariance = curve_fit(sigmoid, delay[0:11], unp.nominal_values(hitrate[0:11]), p0=[0.5, 0.1, 10], sigma=unp.std_devs(hitrate[0:11]))
params2, covariance2 = curve_fit(sigmoid, delay[15:30], unp.nominal_values(hitrate[15:30]), p0=[60, -0.16, 10],sigma=unp.std_devs(hitrate[15:30]))

errors = np.sqrt(np.diag(covariance))
errors2 = np.sqrt(np.diag(covariance2))

print('Parameter:', params)
print('Parameter2:', params2)
print('Fehler:', errors)
print('Fehler2:', errors2)

x = np.linspace(-16, -6, 1000)
x2 = np.linspace(1, 12, 1000)

fit = sigmoid(x, *params)
fit2 = sigmoid(x2, *params2)

idx = (np.abs(fit - fwhm_y)).argmin()
idx2 = (np.abs(fit2 - fwhm_y)).argmin()

fwhm_x = x[idx]
fwhm_x2 = x2[idx2]

print('FWHM x:', fwhm_x)
print('FWHM x2:', fwhm_x2)

fwhm_breite = fwhm_x2 - fwhm_x

diskriminator_breite = unc.ufloat(12, 0.5)

aufloesung = np.abs(2*diskriminator_breite - fwhm_breite)

print('Auflösung:', aufloesung)
print('FWHM Breite:', fwhm_breite)
print(fwhm_y)

plt.axvline(x=fwhm_x, color='green', linestyle='--', label='FWHM', alpha=0.7)
plt.axvline(x=fwhm_x2, color='green', linestyle='--', alpha=0.7)

plt.plot(x, fit, color='green', label='Fit mit Sigmoidfunktion', alpha=0.5)
plt.plot(x2, fit2, color='green', alpha=0.5)

plt.legend(loc='best', fontsize='x-small', markerscale=0.5)

plt.savefig('build/delay.pdf')