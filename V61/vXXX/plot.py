import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from uncertainties import ufloat
from scipy.optimize import curve_fit

#Daten Einlesen
res_len_1, P_1  = np.loadtxt('stab_bed_1.txt', delimiter=',', unpack = True)

#Rechnen
#Gitter
dist = ufloat(60, 0.5)
grid1_dist_l = np.array([4.2, 7.6, 11.7, 16.2, 20.9, 25.8, 31.5, 37.9]) *10
grid1_dist_r = np.array([4.2, 7.6, 11.5, 15.3, 19.2, 23.4, 27.9, 32.7]) *10
grid2_dist_l = np.array([3.2, 6.2, 9.6, 13, 16.8, 20.8, 24.5, 29.4, 34.3]) *10
grid2_dist_r = np.array([3, 6, 9, 12.1, 15.1, 18.2, 21.4, 24.7, 28.2]) *10
grid3_dist_l = np.array([12.2, 34.1]) *10
grid3_dist_r = np.array([12.4, 36.1]) *10
grid4_dist_l = np.array([33.4]) *10
grid4_dist_r = np.array([35.2]) *10
g1 = 100
g2 = 80
g3 = 600
g4 = 1200
d1 = ufloat(60, 0.5) *10
d2 = ufloat(60, 0.5) *10
d3 = ufloat(30, 0.5) *10
d4 = ufloat(30, 0.5) *10
dist_1 = unp.uarray([(grid1_dist_l[i] + grid1_dist_r[i]) / 2 for i in range(len(grid1_dist_l))], [np.std([grid1_dist_l[i], grid1_dist_r[i]], ddof=1) for i in range(len(grid1_dist_l))])
dist_2 = unp.uarray([(grid2_dist_l[i] + grid2_dist_r[i]) / 2 for i in range(len(grid2_dist_l))], [np.std([grid2_dist_l[i], grid2_dist_r[i]], ddof=1) for i in range(len(grid2_dist_l))])
dist_3 = unp.uarray([(grid3_dist_l[i] + grid3_dist_r[i]) / 2 for i in range(len(grid3_dist_l))], [np.std([grid3_dist_l[i], grid3_dist_r[i]], ddof=1) for i in range(len(grid3_dist_l))])
dist_4 = unp.uarray([(grid4_dist_l[i] + grid4_dist_r[i]) / 2 for i in range(len(grid4_dist_l))], [np.std([grid4_dist_l[i], grid4_dist_r[i]], ddof=1) for i in range(len(grid4_dist_l))])

def W(dists, g , L):
    lam = unp.uarray([((g * dists[i])/((i+1) * unp.sqrt(L**2 + dists[i]**2))).nominal_value for i in range(len(dists))],[((g * dists[i])/((i+1) * unp.sqrt(L**2 + dists[i]**2))).std_dev for i in range(len(dists))]) 
    return np.sum(lam)/len(lam) * 1e6

print('Für Grid 1 beträgt die bestimmte Wellenlänge',W(dist_1, 1/g1, d1))
print('Für Grid 2 beträgt die bestimmte Wellenlänge',W(dist_2, 1/g2, d2))
print('Für Grid 3 beträgt die bestimmte Wellenlänge',W(dist_3, 1/g3, d3))
print('Für Grid 4 beträgt die bestimmte Wellenlänge',W(dist_4, 1/g4, d4))

#Plotten
#stab bed
ax, fig = plt.subplots(constrained_layout=True)
fig.scatter(res_len_1, P_1, color='red', label='Datenpunkte', marker='x')
fig.set_xlabel(r'Resonatorlänge / $\mathrm{cm}$')
fig.set_ylabel(r'Leistung / $\mathrm{mW}$')
fig.grid()
ax.legend()
ax.savefig('plots/stab_bed_1.png')
ax.clf()
#polarisation

def pol(deg, phi_0, I_0):
    return I_0 * np.cos((deg - phi_0)/180 * np.pi)**2

x_run = np.linspace(0, 360, 1000)
Deg = np.arange(0, 360, 10)
pol_int = np.loadtxt('pol.txt', unpack = True)
params, covariance = curve_fit(pol, Deg, pol_int, p0=[0, 1])
ax, fig = plt.subplots(constrained_layout=True)
fig.scatter(Deg, pol_int, color='red', marker = 'x', label='Datenpunkte')
fig.plot(x_run, pol(x_run, *params), color='blue', label='Fit')
fig.set_xlabel(r'Polarisationswinkel $^\circ$')
fig.set_ylabel(r'Leistung / $\mathrm{mW}$')
fig.grid()
ax.legend()
ax.savefig('plots/pol.png')
print(covariance)

