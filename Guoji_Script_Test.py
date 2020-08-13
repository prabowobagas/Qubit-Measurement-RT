import numpy as np
import qcodes
import matplotlib.pyplot as plt
from functools import partial    
from  qcodes.plots.qcmatplotlib import MatPlot
from  qcodes.plots.pyqtgraph import QtPlot
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, ConstantModel, GaussianModel

def fit_s21mag(x_val, y_val):
    peak = GaussianModel()
    offset = ConstantModel()
    model = peak + offset
    pars = offset.make_params(c=np.median(y_val))
    pars += peak.guess(y_val, x=x_val, amplitude=-0.5)
    result = model.fit(y_val, pars, x=x_val)
    return result

def fit_loglog_SNR(int_time, SNR, t_int_linspace):
    m, c = np.polyfit(np.log(int_time), np.log(SNR), 1) # fit log(y) = m*log(x) + c
    loglogfit = t_int_linspace**(m) * np.exp(c)
    t_SNR40 = np.exp(-c/m) 
    return [m, c, loglogfit, t_SNR40]


# def SNR_plot(x_val, y_val)

# %% Plotting stuff

int_time = [1.28e-6, 256e-6]
data_1 = pd.read_csv('Guoji Data/20-55-01_qtt_scan1D/RP.dat', delimiter='\t', skiprows = [1, 2]) # 1.28us
data_2 = pd.read_csv('Guoji Data/21-08-15_qtt_scan1D/RP.dat', delimiter='\t', skiprows = [1, 2]) # 256 us


s21mag_1 = data_1['S21mag'].to_numpy()
s21mag_2 = data_2['S21mag'].to_numpy()
s21mag = [s21mag_1, s21mag_2]

vrp = data_1['# RP']


gauss_fit_1 = fit_s21mag(vrp, s21mag_1)
gauss_fit_2 = fit_s21mag(vrp, s21mag_2)
gauss_fit = [gauss_fit_1, gauss_fit_2]
    

t_int_linspace = np.logspace(-6, -3, 50)

S = []
N = []


for i in range(0, len(int_time)):
    S.append(abs(gauss_fit[i].values['height']))
    N.append(np.std(s21mag[i][0:10]))
    SNR = np.power([x / y for x, y in zip(S, N)],2)
    
print(SNR)

m, c = np.polyfit(np.log10(int_time), np.log10(SNR), 1) # fit log(y) = m*log(x) + c

loglogfit = t_int_linspace**(m) * np.exp(c)
t_SNR40 = np.exp(-c/m) 
print(t_SNR40)

plt.figure()
# plt.plot(vrp, gauss_fit_1.best_fit, 'b--')
# plt.plot(s21mag_1)
plt.plot(gauss_fit_2.best_fit, 'b--')
# plt.plot(vrp, s21mag_2)

plt.figure()
plt.loglog(t_int_linspace, loglogfit )
plt.scatter(int_time, SNR)
# # fig = plt.figure()
# # ax = plt.gca()
# # plt.plot(int_time, y_fit, ':')
# # plt.loglog(int_time, SNR)
# # plt.scatter(int_time, SNR)
# # ax.ylim([1e0, 1e4])



