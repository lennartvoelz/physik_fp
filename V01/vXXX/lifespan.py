import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mca


counts = np.loadtxt('../data/myonen.txt', unpack=True)

bins = np.arange(0,512,1)

time = mca.converter(bins)

num_bins = 25

counts, bin_edges = np.histogram(time, bins=num_bins, weights=counts)

counts_err = np.sqrt(counts)

plt.clf()
plt.bar(bin_edges[:-1], counts, align="center", yerr=counts_err, capsize=3, ecolor='black', label="Messwerte", edgecolor='black', linewidth=0.7, width=0.44)

def fit_func(x, a, b, c, d):
    return a*np.exp(-b*(x-c))+d

popt, pcov = curve_fit(fit_func, bin_edges[:-1], counts, p0=[1200, 0.1, 0.1, 0.1])

errors = np.sqrt(np.diag(pcov))

x = np.linspace(np.min(time), np.max(time), 1000)

plt.plot(x, fit_func(x, *popt), label="Fit", color='red')

plt.savefig("build/lifespan_hist.pdf")

print("a = ", popt[0], ", b = ", popt[1], ", c = ", popt[2], ", d = ", popt[3])
print("errors: ", errors)

