from  qcodes.plots.qcmatplotlib import MatPlot
from  qcodes.plots.pyqtgraph import QtPlot
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, ConstantModel, GaussianModel
import numpy as np

def vec2mat(a, new_shape):

    padding = (new_shape - np.vstack((new_shape, a.shape))).T.tolist()
    print(padding)

    return np.pad(a, np.abs(padding), mode='constant')

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

plt.close("all")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2



data = pd.read_csv(
    'Scripts and PPT Summary/CryoRX/2020-06-22/14-54-05_qtt_scan2Dresonatorfast/LP1_RP1.dat',
    skiprows=[0, 2], delimiter='\t'
    ) # Cryo RX Stability diagram

# data = pd.read_csv(
#     'Scripts and PPT Summary/CryoRX/2020-06-22/15-55-00_qtt_scan2Dresonatorfast/LP1_RP1.dat',
#     skiprows=[0, 2], delimiter='\t'
#     ) # Cryo RX Transition zoomed

pin = -40
vin_peak = 10 ** ((pin - 10) / 20) 


data_array = data.to_numpy()

rp = data_array[0:200, 1]

lp = data_array[:, 0]
lp = lp[::200]


amp = 20*np.log10((np.resize(data_array[:,4], (200,200))) / vin_peak) # For Full stability diagram
# amp = 20*np.log10((np.resize(data_array[:,4], (50,200))) / vin_peak) # For zoomed in


plt.figure(figsize=(6, 6), dpi=150)
plt.imshow(amp, origin='lower', extent=[min(rp) , max(rp), min(rp) ,  max(rp)])
plt.colorbar(fraction=0.046, pad=0.04)


plt.xlabel('V$_{RP}$')
plt.ylabel('V$_{LP}$') 
