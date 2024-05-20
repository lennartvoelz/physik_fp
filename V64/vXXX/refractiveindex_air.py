import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import ufloat


p, count1, count2, count3, count4, count5 = np.loadtxt("../data/counts_pump.csv", unpack=True, delimiter=",")
p = p*100

count = np.mean([count1, count2, count3, count4, count5], axis=0)
count_err = np.std([count1, count2, count3, count4, count5], axis=0)

counts = unp.uarray(count, count_err)
print(counts)


lam = 633e-9
L = ufloat(100e-3, 0.1e-3)

n = (counts*lam)/L + 1


def linear(x, a, b):
    return a*x + b

from scipy.optimize import curve_fit

popt, pcov = curve_fit(linear, p, unp.nominal_values(n), sigma=unp.std_devs(n))
popt_err = np.sqrt(np.diag(pcov))

x = np.linspace(0, max(p), 1000)

plt.plot(x, linear(x, *popt), label="Fit", alpha=0.8)
plt.errorbar(p, unp.nominal_values(n), yerr=unp.std_devs(n), fmt='r.', ecolor='black', capsize = 2, capthick=1, label='Calculated refractive index with error')
plt.xlabel(r'$p$ [bar]')
plt.ylabel(r'$n(p)$')
plt.legend()
plt.tight_layout()


print("a = ", popt[0], "+-", popt_err[0])
print("b = ", popt[1], "+-", popt_err[1])


def lorentz_lorenz(p, A, R, T):
    return (3*A*p)/(2*R*T) + 1

T = ufloat(273.15+20.4, 0.1)
R = 8.314
A_oxygen = 4.000
A_nitrogen = ufloat(4.369,0.003)

A = (0.21*A_oxygen + 0.79*A_nitrogen)*1e-6 # inm^3/mol

p1 = np.linspace(0, max(p), 1000)

slope = 3*A/(R*T)
print(slope)

fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].plot(p1, lorentz_lorenz(p1, A.nominal_value, R, T.nominal_value), label="Lorentz-Lorenz", linestyle="--", color="green", alpha=0.7)
ax[0].errorbar(p, unp.nominal_values(n), yerr=unp.std_devs(n), fmt='r.', ecolor='black', capsize = 2, capthick=1, label='Calculated refractive index with error')
ax[0].set_xlabel(r'$p\,/\,$Pa')
ax[0].set_ylabel(r'$n(p)$')
ax[0].legend()

ax[1].plot(x, linear(x, *popt), label="Fit", linestyle="--", color="blue", alpha=0.7)
ax[1].errorbar(p, unp.nominal_values(n), yerr=unp.std_devs(n), fmt='r.', ecolor='black', capsize = 2, capthick=1, label='Calculated refractive index with error')
ax[1].set_xlabel(r'$p\,/\,$Pa')
ax[1].set_ylabel(r'$n(p)$')
ax[1].legend()

plt.tight_layout()
plt.savefig("../build/air_and_lorentz.pdf")