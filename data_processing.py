import numpy as np
from tqdm import tqdm_notebook as tqdm
import qcodes
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, ConstantModel, GaussianModel

class AlazarTech():
    def __init__(self,param):
        self.name = param.get('name')
        self.fs = param.get('sampling_rate')
        self.ts = 1/self.fs
        self.record_length = param.get('record_length') # length in seconds
        self.channel_range = param.get('channel_range')
        self.demod_frequency = param.get('demodulation_frequency')
        self.int_time = param.get('integrate_time')
        self.samples_per_record = int(self.int_time * self.fs) # number of samples in the record length
        
        self.cos_list = []
        self.sin_list = []
        
    def LO_prepare(self):
        integer_list = np.arange(self.samples_per_record)  # Integer list from number of samples recorded per channel
        angle_list = (2 * np.pi * self.demod_frequency * (integer_list / self.fs))
        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)
    
    
    def post_process_data(self, data):
        
        recordA = np.zeros(self.samples_per_record)
        for i in range(0, self.samples_per_record-1):
            recordA[i] = data[i]
            
        recordB = np.zeros(self.samples_per_record)
        for i in range(0, self.samples_per_record-1):
            i0 = i + self.samples_per_record
            recordB[i] = data[i0] 

        s21_demod = self.demodulate_data(recordA, recordB)
        
        s21 = np.average(s21_demod)
        s21m = np.abs(s21)
        s21mag = self.bit2volt(s21m + 127.5)
        
        s21mag_rms = np.sqrt(np.mean(self.bit2volt(np.abs(s21_demod) + 127.5)**2))
        
        s21phase = np.angle(s21, deg=True)
        
        I = self.bit2volt(s21.real + 127.5)
        Q = self.bit2volt(s21.imag + 127.5)
        return [s21mag, s21phase, I, Q, s21mag_rms]
    
    
    def bit2volt(self, signal):
        return (((signal - 127.5) / 127.5) * self.channel_range)
    
    
    def demodulate_data(self, dataA, dataB):
        self.LO_prepare()
        I = (self.cos_list * (dataA - 127.5) + self.sin_list * (dataB - 127.5))
        Q = (self.sin_list * (dataA - 127.5) - self.cos_list * (dataB - 127.5))
        return I+1.j*Q
    


def calc_1Dresonator(Pin, Alazar_obj):
    data = None
    output = pd.DataFrame(data, columns = ['s21mag', 's21phase', 'I', 'Q', 's21mag_rms'])
    for i in range(1,101):
        data_x = pd.read_csv('Data\CryoRX\Pin=' + str(Pin) + 'dBm\Data (' + str(i) + ')', header = None)
        data_x_array = data_x.to_numpy()
        s21mag, s21phase, I, Q, s21mag_rms = Alazar_obj.post_process_data(data_x_array)
        new_row = {'s21mag' : s21mag, 's21phase' : s21phase, 'I' : I, 'Q' : Q, 's21mag_rms' : s21mag_rms}
        output = output.append(new_row,  ignore_index=True)
    return output

def calc_SNR(x_val, y_val, s21mag_rms):
    peak = LorentzianModel()
    offset = ConstantModel()
    model = peak + offset
    
    pars = offset.make_params(c=np.median(y_val))
    pars += peak.guess(y_val, x=x_val, amplitude=-0.5)
    result = model.fit(y_val, pars, x=x_val)
    SNR = np.abs((result.values['height'])**2)/s21mag_rms**2
    return SNR


#------------------------------------
# define setting for ADC
#------------------------------------

adc_param_CryoRX = {
    'name' : 'CryoRX', # For setting ...
    'sampling_rate' : 1e9, # samples/s
    'record_length' : 0.001, # in (s)
    'channel_range' : 0.1, # in (v)
    'demodulation_frequency' : 100e6, # in (Hz)
    'integrate_time' : 0.001 # Must be less than 1ms for our case (Data limited)
    }


adc_param_RT = {
    'name' : 'RT Rack', # For setting ...
    'sampling_rate' : 1e9, # samples/s
    'record_length' : 0.001, # in (s)
    'channel_range' : 0.04, # in (v)
    'demodulation_frequency' : 10e6, # in (Hz)
    'integrate_time' : 0.001
    }


#------------------------------------
# Main
#------------------------------------

if __name__ == "__main__": 
    v_rp1_cryoRX = pd.read_csv(
        'Scripts and PPT Summary/CryoRX/2020-06-22/18-15-29_qtt_scan1D/RP1.dat',
        skiprows=[0, 2], delimiter='\t') # Take care of the DC axis, different per measurement
    
    # v_rp1_rt_rack = pd.read_csv(Data)


    int_time_output = []
    int_time = [0.001]
    # for i in int_time: 
    #     adc_param_CryoRX['integrate_time'] = i
    #     alazar_cryoRX = AlazarTech(adc_param_CryoRX)
    #     output = calc_1Dresonator(-40, alazar_cryoRX)
    #     int_time_output.append(output)
    
    # For RT rack
    # int_time_output = []
    # int_time = [0.001, 0.0001]
    for i in int_time: 
        adc_param_RT['integrate_time'] = i
        alazar_RT = AlazarTech(adc_param_RT)
        output = calc_1Dresonator(-40, alazar_RT)
        int_time_output.append(output)
    

    

    plt.plot(v_rp1_cryoRX['# "RP1"'], int_time_output[0]["s21mag"]) 
    plt.plot(v_rp1_cryoRX['# "RP1"'], int_time_output[1]["s21mag"])
    plt.show()
    
