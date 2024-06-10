import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from uncertainties import unumpy


R1, R2, t, I, U = np.loadtxt("data/measurement.csv", delimiter=",", unpack=True)

def temperture(R):
    return 0.00134 * R**2 + 2.296 * R - 243.02 + 273.15

R1 = unumpy.uarray(R1, 1)
R2 = unumpy.uarray(R2, 1)
T1 = temperture(R1)
T2 = temperture(R2)
t = unumpy.uarray(t, 0.1)
I = unumpy.uarray(I, 0.1) * 10**(-3) # mA -> A
U = unumpy.uarray(U, 0.01)

E = U * I * t

delta_T = T2 - T1

M = 63.546 * 10**(-3) # kg/mol
m = 0.342 # kg

C_p = M/m * E / delta_T

#print("C_p = ", C_p)
#print(delta_T)

kappa = 140 * 10**9 # GPa -> Pa
alpha = np.array([8.5, 9.75, 10.70, 11.50, 12.10, 12.65, 13.15, 13.60, 13.90, 14.25, 14.50, 14.75, 14.95, 15.20, 15.40, 15.60, 15.75, 15.90, 16.10]) * 10**(-6) # 1/K
V_0 = 7.11 * 10**(-6) # m^3/mol

C_V = C_p - 9 * alpha**2 * kappa * T1 * V_0

#print("C_V = ", C_V, "\n\n")
#print(T1, "\n\n")
#print(alpha, "\n\n")

plt.plot(unumpy.nominal_values(T1), unumpy.nominal_values(C_V), "rx", label="C_V")
plt.plot(unumpy.nominal_values(T1), unumpy.nominal_values(C_p), "bx", label="C_p")
plt.xlabel(r"$T$ / K")
plt.ylabel(r"$C$ / J/K")
plt.legend(loc="best")
plt.savefig("build/plot.pdf")
plt.clf()

C_V = C_V[unumpy.nominal_values(T1) < 180]
T1 = T1[unumpy.nominal_values(T1) < 180]

debye = np.array([3.5, 3.1, 2.9, 2.5, 2.4, 2.4, 1.7, 1.8, 2.4, 2.4])
print(debye, "\n\n")

debye = debye * T1

print(debye,"\n\n")
print(T1,"\n\n")
print(C_V,"\n\n")

x = np.array([290.28891009999984, 288.82672721859984, 299.4539636223999, 283.19728183999985, 
              295.85530595999984, 319.8372230399999, 243.5345910021999, 275.8369977791999, 
              391.74300731039995, 415.6929926399999])

sigma = np.array([8.248925999999999, 7.34216524, 6.90150816, 5.977715999999999, 5.7654288, 
                  5.7921216, 4.1215690799999996, 4.38384096, 5.87142816, 5.8976064])

# Berechnung der Gewichte
weights = 1 / sigma**2

# Gewichteter Mittelwert
weighted_mean = np.sum(x * weights) / np.sum(weights)

# Unsicherheit des gewichteten Mittelwerts
weighted_mean_sigma = np.sqrt(1 / np.sum(weights))

print(f"Gewichteter Mittelwert: {weighted_mean:.2f}")
print(f"Unsicherheit des gewichteten Mittelwerts: {weighted_mean_sigma:.2f}")