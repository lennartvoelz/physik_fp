import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

channel, time = np.loadtxt('../data/mca.csv', unpack=True, delimiter=',')

plt.xlabel('Channel')
plt.ylabel(r'Zeit [$\mu s$]')
plt.legend()
plt.grid(True)

def linear_fit(x, a, b):
    return a*x + b

params, covariance = curve_fit(linear_fit, channel, time)
errors = np.sqrt(np.diag(covariance))

print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

x = np.linspace(0, 512, 1000)

plt.plot(x, linear_fit(x, *params), 'b-', label='Lineare Regression')
plt.plot(channel, time, 'rx', label='Messwerte')
plt.legend()

def converter(channel):
    '''
    Converts channel number to time in microseconds.

    Args: numpy array, (int)

    Returns: numpy array, (int)
    '''
    return linear_fit(channel, *params)

plt.savefig('build/mca.pdf')