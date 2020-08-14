from  qcodes.plots.qcmatplotlib import MatPlot
from  qcodes.plots.pyqtgraph import QtPlot
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, ConstantModel, GaussianModel


def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

data = pd.read_csv(
    'Scripts and PPT Summary/RT Setup/2020-06-22/12-13-35_qtt_scan2Dresonatorfast/')

plt.imshow(data, aspect='auto', interpolation='none',
           extent=extents(x) + extents(y), origin='lower')




