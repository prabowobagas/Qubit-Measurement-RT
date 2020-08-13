import numpy as np
from scipy import stats
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
        
        self.IQ_data = []
        
    def LO_prepare(self):
        integer_list = np.arange(self.samples_per_record)  # Integer list from number of samples recorded per channel
        angle_list = (2 * np.pi * self.demod_frequency * (integer_list / self.fs))
        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)
      
    def bit2volt(self, signal):
        return (((signal - 127.5) / 127.5) * self.channel_range)
    
    
    def demodulate_data(self, dataA, dataB):
        self.LO_prepare()
        I = (self.cos_list * (dataA - 127.5) + self.sin_list * (dataB - 127.5))
        Q = (self.sin_list * (dataA - 127.5) - self.cos_list * (dataB - 127.5))
        I_volt = self.bit2volt(I + 127.5)
        Q_volt = self.bit2volt(Q + 127.5)
        return I_volt+1.j*Q_volt
        
    def calc_RMS(self, signal):
        rms = np.sqrt(np.mean(np.power(signal,2)))
        return rms
        
    def post_process_data(self, data):
        
        recordA = np.zeros(self.samples_per_record)
        for i in range(0, self.samples_per_record-1):
            recordA[i] = data[i]
            
        recordB = np.zeros(self.samples_per_record)
        for i in range(0, self.samples_per_record-1):
            i0 = i + self.samples_per_record
            recordB[i] = data[i0] 

        IQ_volt = self.demodulate_data(recordA, recordB)
        
        IQ_avg = np.mean(IQ_volt)
        IQ_mag = np.abs(IQ_avg)
        
        IQ_rms = np.sqrt(np.std(IQ_volt.real)**2 + np.std(IQ_volt.imag)**2)

        
        IQ_phase = np.angle(IQ_avg, deg=True)
        
        I = IQ_avg.real
        Q = IQ_avg.imag
        return [IQ_mag, IQ_phase, I, Q, IQ_rms]


def calc_1Dresonator(Pin, Alazar_obj, en):
    data = None
    output = pd.DataFrame(data, columns = ['s21mag', 's21phase', 'I', 'Q', 's21mag_rms'])
    for i in range(1,101):
        if en == True:
            df_data = pd.read_csv('Data\CryoRX\Pin=' + str(Pin) + 'dBm\Data (' + str(i) + ')', header = None)
        elif en == False:
            df_data = pd.read_csv('Data\Standard Setup\Pin=' + str(Pin) + 'dBm\Data (' + str(i) + ')', header = None)

        data_array = df_data.to_numpy()
        s21mag, s21phase, I, Q, s21mag_rms = Alazar_obj.post_process_data(data_array)
        new_row = {'s21mag' : s21mag, 's21phase' : s21phase, 'I' : I, 'Q' : Q, 's21mag_rms' : s21mag_rms}
        output = output.append(new_row,  ignore_index=True)
        
        if i % 50 == 0:
            print(i)
            
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

def fit_s21mag(x_val, y_val):
    peak = GaussianModel()
    offset = ConstantModel()
    model = peak + offset
    pars = offset.make_params(c=np.median(y_val))
    pars += peak.guess(y_val, x=x_val, amplitude=-0.5)
    result = model.fit(y_val, pars, x=x_val)
    return result


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
        skiprows=[0, 2], delimiter='\t'
        ) # Take care of the DC axis, different per measurement
    
    # v_rp1_rt_rack = pd.read_csv(Data)


    int_time_output = []
    # int_time = np.logspace(np.log10(1e-5), np.log10(1e-3), 10)
    # int_time =  np.linspace(1e-5, 1e-3), 10)
    # int_time = [10e-6, 20e-6, 40e-6, 80e-6, 160e-6, 320e-6, 640e-6, 1000e-6]
    int_time = np.linspace(1e-5, 1e-7, 10)

    enable = False
    
    for i in int_time: 

        if enable == True:
            adc_param_CryoRX['integrate_time'] = i
            alazar_cryoRX = AlazarTech(adc_param_CryoRX)
            output = calc_1Dresonator(-40, alazar_cryoRX, True)
            int_time_output.append(output)
        elif enable == False:
            adc_param_RT['integrate_time'] = i
            alazar_RT = AlazarTech(adc_param_RT)
            output = calc_1Dresonator(-40, alazar_RT, False)
            int_time_output.append(output)
        

                
    

# %% SNR 

pin = -40
vin_peak = 10 ** ((pin - 10) / 20) 

SNR = []
S = []
N = []

for idx,output in enumerate(int_time_output):
    v_rp = v_rp1_cryoRX['# "RP1"']
    s21 = output["s21mag"] / vin_peak # convert s21mag (digitzed)
    
    # plt.plot(s21) 
    gauss_fit = fit_s21mag(v_rp, s21)
    S.append(abs(gauss_fit.values['height']))
    N.append(np.std(s21[0:15]))
    SNR.append((S[idx] / N[idx])**2)
    


t_int_linspace = np.linspace(5e-7, 5e-3, 1000)
m, c = np.polyfit(np.log(int_time), np.log(SNR), 1) # fit log(y) = m*log(x) + c
loglogfit = t_int_linspace**(m) * np.exp(c)
t_min = np.exp(-c / m) 

plt.figure(figsize=(8,3), dpi=150)


plt.scatter(int_time, SNR)
plt.loglog(t_int_linspace, loglogfit, label='RT Rack Setup (11 MHz)')
plt.legend()
plt.text(1e-3, 8e-1, ' Pin = -40 dBm \n t$_{min}$='+str(round(t_min,2))+'$\mu$S', fontsize=12)
plt.xlabel('t$_{int}$')
plt.ylabel('SNR (a.u)') 
# ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
# fig = plt.figure()
# ax = plt.gca()
# plt.plot(int_time, y_fit, ':')
# plt.loglog(int_time, SNR)
# plt.scatter(int_time, SNR)
# ax.ylim([1e0, 1e4])

    