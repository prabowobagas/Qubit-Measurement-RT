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

class AlazarTech():
    def __init__(self,param):
        self.name = param.get('name')
        self.fs = param.get('sampling_rate')
        self.ts = 1/self.fs
        self.record_length = param.get('record_length') # length in seconds
        self.channel_range = param.get('channel_range')
        self.demod_frequency = param.get('demodulation_frequency')
        self.int_time = param.get('integrate_time')
        self.samples_per_record = int(self.record_length * self.fs) # number of samples in the record length
        
        self.cos_list = []
        self.sin_list = []
        
    def LO_prepare(self):
        integer_list = np.arange(round(self.record_length * self.fs))  # Integer list from number of samples recorded per channel
        angle_list = (2 * np.pi * self.demod_frequency * (integer_list / self.fs))
        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)
    
    
    def post_process_data(self, data):
        
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.samples_per_record):
            recordA[i] = data[i]
            
        recordB = np.zeros(self.samples_per_record)
        for i in range(self.samples_per_record):
            i0 = i + self.samples_per_record // 2
            recordB[i] += data[i0] 
        
        
        s21 = self.demodulate_data(recordA, recordB)
        s21m = np.abs(s21)
        s21mag = self.bit2volt(s21m + 127.5)
        s21phase = np.angle(s21, deg=True)
        I = self.bit2volt(s21.real + 127.5)
        Q = self.bit2volt(s21.imag + 127.5)
    
        return [s21mag, s21phase, I, Q]
    
    def bit2volt(self, signal):
        return (((signal - 127.5) / 127.5) * self.channel_range)
    
    def demodulate_data(self, dataA, dataB):
        self.LO_prepare()
        I = np.average(self.cos_list * (dataA - 127.5) + self.sin_list * (dataB - 127.5))
        Q = np.average(self.sin_list * (dataA - 127.5) - self.cos_list * (dataB - 127.5))
        return I+1.j*Q
    
def CryoRX_setup():
    data = None
    output = pd.DataFrame(data, columns = ['s21mag', 's21phase', 'I', 'Q'])
    for i in range(1,100):
        data_x = pd.read_csv('Data\CryoRX\Pin=-40dBm\Data (' + str(i) + ')')
        data_x_array = data_x.to_numpy()
        s21mag, s21phase, I, Q = alazar_cryoRX.post_process_data(data_x_array)
        new_row = {'s21mag' : s21mag, 's21phase' : s21phase, 'I' : I, 'Q' : Q}
        output = output.append(new_row,  ignore_index=True)
    return output

def RTrack_setup():
    data = None
    output = pd.DataFrame(data, columns = ['s21mag', 's21phase', 'I', 'Q'])
    for i in range(1,100):
        data_x = pd.read_csv('Data\CryoRX\Pin=-40dBm\Data (' + str(i) + ')')
        data_x_array = data_x.to_numpy()
        s21mag, s21phase, I, Q = alazar_RT.post_process_data(data_x_array)
        new_row = {'s21mag' : s21mag, 's21phase' : s21phase, 'I' : I, 'Q' : Q}
        output = output.append(new_row,  ignore_index=True)
    return output






#------------------------------------
# define setting for ADC
#------------------------------------

adc_param_CryoRX = {
    'name' : 'CryoRX', # For setting ...
    'sampling_rate' : 1e9, # samples/s
    'record_length' : 0.001, # in (s)
    'channel_range' : 0.1, # in (v)
    'demodulation_frequency' : 100e6, # in (Hz)
    'integrate_time' : 0.001
    }


adc_param_RT = {
    'name' : 'CryoRX', # For setting ...
    'sampling_rate' : 1e9, # samples/s
    'record_length' : 0.001, # in (s)
    'channel_range' : 0.02, # in (v)
    'demodulation_frequency' : 10e6 # in (Hz)
    }


#------------------------------------
# Main
#------------------------------------

if __name__ == "__main__": 
    alazar_cryoRX = AlazarTech(adc_param_CryoRX)
    alazar_RT = AlazarTech(adc_param_RT)
    
    output = CryoRX_setup()
    plt.plot(output["s21mag"])