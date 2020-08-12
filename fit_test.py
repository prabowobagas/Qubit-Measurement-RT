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



pd_dat = pd.read_csv(
    'Scripts and PPT Summary/CryoRX/2020-06-22/18-15-29_qtt_scan1D/RP1.dat',
    skiprows=[0, 2], delimiter='\t')

xval = pd_dat['# "RP1"']
yval = pd_dat['S21mag']


peak = GaussianModel()
offset = ConstantModel()
model = peak + offset

pars = offset.make_params(c=np.median(yval))
pars += peak.guess(yval, x=xval, amplitude=-0.5)
result = model.fit(yval, pars, x=xval)

x = [[1+1.j*1], [1+1.j*100]]
print (np.mean(x))
print(abs(result.values['height']))

plt.plot(xval, yval)
plt.plot(xval, result.best_fit, 'b--')
plt.show()