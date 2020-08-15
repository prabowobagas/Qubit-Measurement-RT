from  qcodes.plots.qcmatplotlib import MatPlot
from  qcodes.plots.pyqtgraph import QtPlot
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, ConstantModel, GaussianModel
import numpy as np




def fit_s21mag(x_val, y_val):
    peak = LorentzianModel()
    offset = ConstantModel()
    model = peak 
    pars = peak.guess(y_val, x=x_val, amplitude=-0.05)
    result = model.fit(y_val, pars, x=x_val)
    return result


plt.close("all")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2


col = plt.cm.binary(np.linspace(0.1, 0.6, 1))


data = pd.read_csv(
    'Scripts and PPT Summary/CryoRX/2020-06-22/15-31-49_qtt_RFscan1D/frequency.DAT',
    skiprows=[0, 2], delimiter='\t'
    ) # Take care of the DC axis, different per measurement) # Resonator 

pin = -40
vin_peak = 10 ** ((pin - 10) / 20) 


s21 = data['S21mag'].to_numpy() / vin_peak
freq = data['# "Frequency"'].to_numpy() 
Lorentzfit = fit_s21mag(freq , s21)


plt.figure(figsize=(5,4), dpi=150)
plt.plot(freq, 20*np.log10(s21))

# plt.plot(freq, 20*np.log10(Lorentzfit.best_fit), 'r--', linewidth=2, zorder=11)

plt.grid(color=col[0], linestyle='--', linewidth=2)

plt.xlabel('Frequency (Hz)')
plt.ylabel('S21 (dB)') 
plt.ylim([-45, -10])


















