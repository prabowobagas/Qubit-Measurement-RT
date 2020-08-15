from  qcodes.plots.qcmatplotlib import MatPlot
from  qcodes.plots.pyqtgraph import QtPlot
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, ConstantModel, GaussianModel
import numpy as np


plt.close("all")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

# data = pd.read_csv(
    # 'Scripts and PPT Summary/RT /2020-06-22/12-13-35_qtt_scan2Dresonatorfast/') # RT Setup

data = pd.read_csv(
    'Scripts and PPT Summary/CryoRX/2020-06-22/14-54-05_qtt_scan2Dresonatorfast/LP1_RP1.dat',
    skiprows=[0, 2], delimiter='\t'
    ) # Cryo RX

LP1 = data['# "LP1"'].to_numpy()
RP1 = data['RP1'].to_numpy()
S21mag = data ['S21mag'].to_numpy()

A = [RP1, LP1, S21mag]
plt.imshow(A, aspect='auto', interpolation='none',
            extent=extents(RP1) + extents(LP1), origin='lower')




