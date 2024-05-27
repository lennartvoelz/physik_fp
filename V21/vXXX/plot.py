import numpy as np
import matplotlib.pyplot as plt
from uncertainties.unumpy import uarray
from uncertainties import ufloat
import scipy.constants as const
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp

mu_B = const.physical_constants['Bohr magneton'][0]
mu_0 = const.physical_constants['mag. constant'][0]
h = const.physical_constants['Planck constant'][0]
N =  154
R = 15.79e-2


# Daten einlesen
p1 = np.genfromtxt('data/peak1.txt', unpack=True) /10
p2 = np.genfromtxt('data/peak2.txt', unpack=True) /10
os_1 = np.genfromtxt('data/os_p1.txt', unpack=True)
os_2 = np.genfromtxt('data/os_p2.txt', unpack=True)

def get_I(p, os):
    return p + os * 2 / 1000

def B(I):
    return mu_0 * 8 * I * N / (np.sqrt(125) * R)


I1 = get_I(p1, os_1)
I2 = get_I(p2, os_2)


B1 = B(I1)
B2 = B(I2)

x = np.arange(100000, 1100000, 100000)

def f(x, m, b):
    return m * x + b

#plot
params1, covariance1 = curve_fit(f, x, B1)
errors1 = np.sqrt(np.diag(covariance1))
params2, covariance2 = curve_fit(f, x, B2)
errors2 = np.sqrt(np.diag(covariance2))
print('Peak 1: ', params1, errors1)
print('Peak 2: ', params2, errors2)
print('Erdmagnetfeld vertikal: ', B(0.0244))
x_plot = np.linspace(0, 1100000, 1000)
plt.plot(x_plot/1e6, f(x_plot, *params1)*1e6, 'r--', label='Fit Peak 1')
plt.plot(x_plot/1e6, f(x_plot, *params2)*1e6, 'b--', label='Fit Peak 2')
plt.plot(x/1e6, B1*1e6, 'rx', label='Peak 1')
plt.plot(x/1e6, B2*1e6, 'bx', label='Peak 2')
plt.xlabel(r'Freuquenz $f$ / $\mathrm{Hz}$')
plt.ylabel(r'Magnetfeld $B$ / $\mathrm{\mu T}$')
plt.legend()
plt.savefig('build/plot1.pdf')

# Berechnung der Landé-Faktoren
def gJ(J, S, L):
    return (3*J*(J+1) +  (S*(S+1) - L*(L+1))) / (2*J*(J+1))

def gF(J, S, L, F, I):
    return gJ(J, S, L) * (F*(F+1) + J*(J+1) - I*(I+1)) / (2*F*(F+1))

gF_1_theo = gF(1/2, 1/2, 0, 2, 3/2)
gF_2_theo = gF(1/2, 1/2, 0, 3, 5/2)
m1 = ufloat(params1[0], errors1[0])
m2 = ufloat(params2[0], errors2[0])
gF_1 = h/(mu_B * m1)
gF_2 = h/(mu_B * m2)
print('gF_1: ', gF_1)
print('gF_2: ', gF_2)
print('gF_1_theo: ', gF_1_theo)
print('gF_2_theo: ', gF_2_theo)
print('gF_1_theo/gF_2_theo: ', gF_1_theo / gF_2_theo)
print('gF_1/gF_2: ', gF_1 / gF_2)

#berechnung kernspins

def I(gF, J, S, L, F):
    return -1/2 + unp.sqrt(1/4 + (J*(J+1) + F*(F+1) - gF/gJ(J,S,L) * 2*F*(F+1)) )

I_1 = I(gF_1, 1/2, 1/2, 0, 2)
I_2 = I(gF_2, 1/2, 1/2, 0, 3)
print('I_1: ', I_1)
print('I_2: ', I_2)

#Zeemann Effekt

def delE_zem(gF, B, m, E_Hy):
    return gF * mu_B * B + gF**2 * mu_B**2 * B**2 * (1 - 2 * m) / (E_Hy) * 6.242e+18

E_hy2 = 2.01e-24
E_hy1 = 4.53e-24

delE_zem_1 = delE_zem(gF_1, B1, 2, E_hy1)


# verhältnisse 
len_s = 18
len_l = 34

print('Anteil Element 1', len_s / (len_s + len_l))
print('Anteil Element 2', len_l / (len_s + len_l))




