import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import ufloat

maxima_counts = np.loadtxt('../data/maxima_counts.csv', unpack=True)

n_max = np.mean(maxima_counts)
n_max_err = np.std(maxima_counts)

max_counts = unc.ufloat(n_max, n_max_err)

print("counts + error: ",max_counts)

lam = 633*10**(-9)
theta = np.deg2rad(10)
theta_0 = np.deg2rad(10)
d = 1e-3

def refrac_idx(M):
    return 1/(1-(lam*M)/(2*d*theta*theta_0))

n = refrac_idx(max_counts)

print("refractive index: ",n)