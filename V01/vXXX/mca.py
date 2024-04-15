import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

channel, time = np.loadtxt('mca.csv', unpack=True, delimiter=',')

plt.xlabel('Channel')
plt.ylabel(r'Zeit [$\mu s$]')
plt.title('Zeit vs. Channel')
plt.legend()
plt.grid(True)
plt.gca().set_facecolor('#f7f7f7')

def linear_fit(x, a, b):
    return a*x + b

params, covariance = curve_fit(linear_fit, channel, time)
errors = np.sqrt(np.diag(covariance))

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

plt.savefig('mca.pdf')