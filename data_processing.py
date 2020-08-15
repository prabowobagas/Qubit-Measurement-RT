import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit.models import ConstantModel, GaussianModel

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


def calc_1Dresonator(Pin, Alazar_obj, en, int_time):
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
        if i % 100 == 0:
            print(int_time)
        
    return output


def fit_s21mag(x_val, y_val):
    peak = GaussianModel()
    offset = ConstantModel()
    model = peak + offset
    pars = offset.make_params(c=np.median(y_val))
    pars += peak.guess(y_val, x=x_val, amplitude=-0.05, center= x_val[50])
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

pin = -40
vin_peak = 10 ** ((pin - 10) / 20) 

int_time_output = []
# int_time = [10e-6, 20e-6, 40e-6, 80e-6, 160e-6, 320e-6, 640e-6, 1000e-6]
# int_time = [8e-6, 16e-6, 32e-6, 64e-6, 128e-6, 256e-6, 512e-6, 725e-6, 819e-6, 1000e-6]
int_time = [8e-6, 16e-6, 32e-6, 62e-6, 124e-6, 248e-6, 496e-6, 992e-6] # Good values for SNR calculation
int_time = np.logspace(np.log(50e-6), np.log(992e-6), 20)
# int_time = [5e-6, 25e-6, 50e-6, 100-6, 500e-6, 956e-6]
# int_time = [8e-6, 11e-6, 18e-6, 32e-6, 64e-6, 128e-6, 256e-6, 312e-6, 512e-6, 712e-6,  1000e-6]
# int_time = np.round(np.logspace(np.log10(1e-6), np.log10(1e-3), 5, base=10),6)

enable = True

for idx,i in enumerate(int_time): 

    if enable == True:
        adc_param_CryoRX['integrate_time'] = i
        alazar_cryoRX = AlazarTech(adc_param_CryoRX)
        output = calc_1Dresonator(pin, alazar_cryoRX, True, int_time[idx])
        int_time_output.append(output)
    elif enable == False:
        adc_param_RT['integrate_time'] = i
        alazar_RT = AlazarTech(adc_param_RT)
        output = calc_1Dresonator(pin, alazar_RT, False, int_time[idx])
        int_time_output.append(output)
    

                

# %% SNR Calculation
plt.close("all")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2

n = len(int_time_output) - 1
col_shade = plt.cm.binary(np.linspace(0.1, 0.5, n))
col_last = plt.cm.binary(np.linspace(1, 1, 1))
col = np.concatenate([col_shade, col_last])


SNR = []
S = []
N = []

# DC X-axis data
if enable == True:
    v_rp_cmos  = pd.read_csv(
        'Scripts and PPT Summary/CryoRX/2020-06-22/18-15-29_qtt_scan1D/RP1.dat',
        skiprows=[0, 2], delimiter='\t')
    v_rp = v_rp_cmos['# "RP1"'].to_numpy()
else:
    v_rp_RT = pd.read_csv(
        'Scripts and PPT Summary/RT Setup/2020-06-23/09-28-33_qtt_scan1D/RP1.dat',
    skiprows=[0, 2], delimiter='\t')
    v_rp = v_rp_RT['# "RP1"'].to_numpy()
    

# Plotting the 
plt.figure(figsize=(5,4), dpi=150)
for idx,output in enumerate(int_time_output):
    s21 = output["s21mag"].to_numpy() / vin_peak # convert s21mag (digitzed)
    if idx == len(int_time_output) - 1:
        plt.plot(v_rp, 20*np.log10(s21), linewidth=3, color=col[idx], label='t$_{int}$ = 1 ms \nPin=-40 dBm', zorder=10) 
    else:
        plt.plot(v_rp, 20*np.log10(s21), linewidth=2, color=col[idx]) 
    gauss_fit = fit_s21mag(v_rp, s21)

    S.append(abs(gauss_fit.values['height']))
    N.append(np.std(s21[0:20])) # Noise calculated by the RMS
    SNR.append((S[idx] / N[idx])**2)


plt.plot(v_rp, 20*np.log10(gauss_fit.best_fit), 'r--', linewidth=2, label='Gaussian Fit', zorder=11)

plt.legend()
plt.xlabel('V$_{RP} [mV]$')
plt.ylabel('S21 [dB]') 
# plt.ylim([-30, 0])
plt.grid(color=col[1], linestyle='--', linewidth=2)

t_int_linspace = np.linspace(5e-7, 5e-2, 1000)
m, c = np.polyfit(np.log(int_time), np.log(SNR), 1) # fit log(y) = m*log(x) + c
loglogfit = t_int_linspace**(m) * np.exp(c)
t_min = np.exp(-c / m) 


plt.figure(figsize=(8,4), dpi=150)
plt.scatter(int_time, SNR)
plt.loglog(t_int_linspace, loglogfit, label='CMOS Chip')
plt.legend()
plt.grid(color=col[1], linestyle='--', linewidth=2)
plt.text(1e-4, 1e-1, ' Pin = '+str(pin)+' dBm \n t$_{min}$='+str(round(t_min*1e6,2))+'$\mu$S', fontsize=12)

plt.xlabel('t$_{int}$')
plt.ylabel('SNR (a.u)') 
plt.xlim([2e-6, 2e-3])
# plt.ylim([10e-1, 2e-4])
# %% SNR 

# df = pd.DataFrame([int_time, SNR, t_int_linspace, loglogfit])
# df.to_csv('Data/Processed Data/RT_SNR_9points.csv')
# # df.O

    
