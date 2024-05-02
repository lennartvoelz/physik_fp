import numpy as np
import matplotlib.pyplot as plt
from uncertainties import unumpy as unp
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.constants import c as speed_of_light

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
# ax, fig = plt.subplots(constrained_layout=True)
# fig.scatter(res_len_1, P_1, color='red', label='Datenpunkte', marker='x')
# fig.set_xlabel(r'Resonatorlänge / $\mathrm{cm}$')
# fig.set_ylabel(r'Leistung / $\mathrm{mW}$')
# fig.grid()
# ax.legend()
# ax.savefig('plots/stab_bed_1.png')
# ax.clf()
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
ax.savefig('plots/pol.pdf')
print(f'Phi_0 berechnet sich auf {params[0]} +- {covariance[0][0]} und I_0 berechnet sich auf {params[1]} +- {covariance[1][1]}')


# Fitting models
run_00 = np.linspace(-8, 8, 1000)
def TEM_00(x, I_0, x_1, w):
    return I_0 * np.exp(-2 * (x-x_1)**2 / w**2)
run_01 = np.linspace(-10, 10, 1000)
def TEM_01(x, I_0,x_0, x_1, w):
    return I_0 * ((x-x_0) / w)**2 * np.exp(-2 * (x-x_1)**2 / w**2)

#TEM00 plotting + fitting
dist_00 = np.arange(-8, 9, 1)
dist_01 = np.arange(-10, 11, 1)
P_00 = np.loadtxt('TEM_00.txt', unpack = True)

p_opt, cov = curve_fit(TEM_00, dist_00, P_00, p0=[1, 0, 1])
ax, fig = plt.subplots(constrained_layout = True)
fig.scatter(dist_00, P_00, color='red', marker = 'x', label='Datenpunkte')
fig.plot(run_00, TEM_00(run_00, *p_opt), color='blue', label='Fit')
fig.set_xlabel(r'Abstand von der Strahlachse / $\mathrm{mm}$')
fig.set_ylabel(r'Leistung / $\mathrm{mW}$')
fig.grid()
ax.legend()
ax.savefig('plots/TEM_00.pdf')

print(f'Für den Fit der TEM00 Funktion ergibt sich I_0 = {p_opt[0]} +- {cov[0][0]}, x_1 = {p_opt[1]} +- {cov[1][1]} und w = {p_opt[2]} +- {cov[2][2]}')
#TEM01 plotting + fitting

P_01 = np.loadtxt('TEM_01.txt', unpack = True)

p_opt, cov = curve_fit(TEM_01, dist_01, P_01, p0=[1, 2, 0, 1])
ax, fig = plt.subplots(constrained_layout = True)
fig.scatter(dist_01, P_01, color='red', marker = 'x', label='Datenpunkte')
fig.plot(run_01, TEM_01(run_01, *p_opt), color='blue', label='Fit')
fig.set_xlabel(r'Abstand von der Strahlachse / $\mathrm{mm}$')
fig.set_ylabel(r'Leistung / $\mathrm{mW}$')
fig.grid()
ax.legend()
ax.savefig('plots/TEM_01.pdf')

#print params of TEM_00 and TEM_01 with descriptions
print(f'Für den Fit der TEM01 Funktion ergibt sich I_0 = {p_opt[0]} +- {cov[0][0]}, x_0 = {p_opt[1]} +- {cov[1][1]}, x_1 = {p_opt[2]} +- {cov[2][2]} und w = {p_opt[3]} +- {cov[3][3]}')

#longitudinal modes
lengths = np.array([55.5, 70, 80, 90, 100, 110]) /100
peaks_1 = np.array([266, 533, 799, 1065])
peaks_2 = np.array([218, 435, 653, 818])
peaks_3 = np.array([188, 357, 563, 746, 934])
peaks_4 = np.array([169, 338, 506, 675, 844, 1013, 1181, 1350])
peaks_5 = np.array([154, 304, 454, 608, 758, 908, 1061])
peaks_6 = np.array([139, 274, 409, 544, 679, 814])

del_p1_arr = peaks_1[1:] - peaks_1[:-1]
del_p2_arr = peaks_2[1:] - peaks_2[:-1]
del_p3_arr = peaks_3[1:] - peaks_3[:-1]
del_p4_arr = peaks_4[1:] - peaks_4[:-1]
del_p5_arr = peaks_5[1:] - peaks_5[:-1]
del_p6_arr = peaks_6[1:] - peaks_6[:-1]

del_p = unp.uarray([np.mean(del_p1_arr), np.mean(del_p2_arr), np.mean(del_p3_arr), np.mean(del_p4_arr), np.mean(del_p5_arr), np.mean(del_p6_arr)], [np.std(del_p1_arr, ddof=1), np.std(del_p2_arr, ddof=1), np.std(del_p3_arr, ddof=1), np.std(del_p4_arr, ddof=1), np.std(del_p5_arr, ddof=1), np.std(del_p6_arr, ddof=1)]) * 1e6
calc_len = speed_of_light / (2 * del_p)
print(calc_len)

def fit(x, a, b):
    return a * x + b
run_l = np.linspace(0.53, 1.15, 1000)
p_opt, cov = curve_fit(fit, lengths, unp.nominal_values(calc_len))
print(p_opt)
print(cov)
ax, fig = plt.subplots(constrained_layout = True)
fig.errorbar(lengths, unp.nominal_values(calc_len), yerr=unp.std_devs(calc_len), fmt='o', color='red', ecolor='black', linestyle='', linewidth=1.5, markersize=4, label='Berechnete Resonatorlängen mit Fehler')
fig.plot(run_l, fit(run_l, *p_opt), color='blue', label='Fit')
fig.set_xlabel(r'Resonatorlänge / $\mathrm{m}$')
fig.set_ylabel(r'Berechnete Resonatorlänge / $\mathrm{m}$')
fig.grid()
fig.legend(loc='upper left')
ax.savefig('plots/long_mode.pdf')

#theorie plots stabilitätsbedingung
def stab_1(L):
    return (1- L/1400)**2

def stab_2(L):
    return (1- L/1400)

x_run = np.linspace(0, 2850, 10000)

ax, fig = plt.subplots(constrained_layout = True)
fig.plot(x_run, stab_1(x_run), color='red', label='Stabilitätbedingung Bikonkaver Resonator')
fig.plot(x_run, stab_2(x_run), color='blue', label='Stabilitätsbedingung Plan-Konkaver Resonator')
#mark borders of stability zones for both resonators on the x axis with a vertical line
fig.axvline(x=1400, color='blue', linestyle='--', label = 'Stabilitätsgrenze Bikonkaver Resonator')
fig.axvline(x=2800, color='red', linestyle='--' , label = 'Stabilitätsgrenze Plan-Konkaver Resonator')
fig.fill_between(x_run, 0, 1, color='green', alpha=0.4, label='Stabilitätsbereich')
fig.set_xlabel(r'Resonatorlänge $L$ / $\mathrm{mm}$')
fig.set_ylabel(r'Stabilitätsparameter $g_1 \cdot g_2$')
fig.set_ylim(-0.6, 1.2)
fig.grid()
fig.legend(loc = 'lower left')
ax.savefig('plots/stab_theo.pdf')








