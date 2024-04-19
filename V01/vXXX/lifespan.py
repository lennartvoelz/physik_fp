import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mca
import uncertainties as unc


counts = np.loadtxt('../data/myonen.txt', unpack=True)

bins = np.arange(0,512,1)

time = mca.converter(bins)

num_bins = 25

counts, bin_edges = np.histogram(time, bins=num_bins, weights=counts)

counts_err = np.sqrt(counts)

plt.clf()
plt.bar(bin_edges[:-1], counts, align="center", yerr=counts_err, capsize=3, ecolor='black', label="Messwerte", edgecolor='black', linewidth=0.7, width=0.44)

def fit_func(x, a, b):
    return a*np.exp(-x/b)

popt, pcov = curve_fit(fit_func, bin_edges[:-1], counts, p0=[800, 2.2])

errors = np.sqrt(np.diag(pcov))

x = np.linspace(np.min(time), np.max(time), 1000)


plt.plot(x, fit_func(x, *popt), label="Fit", color='red')
plt.plot(x, fit_func(x, counts[1], 2.197083), label = 'Theorie Kurve', color = 'orange')
plt.xlabel(r'Zeit [$\mu s$]')
plt.ylabel('Counts')
plt.legend(loc='best')

plt.savefig("build/lifespan_hist.pdf")

print("popt: ", popt)
print("errors: ", errors)

binning_breite = bin_edges[1] - bin_edges[0]

print("Breite der Bins: ", binning_breite)

#b = unc.ufloat(popt[1], errors[1])



