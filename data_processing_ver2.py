import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit.models import ConstantModel, GaussianModel
import time

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

def calc_s21vsVrp(int_time, pin, adc_param):
    int_time_output = []
    for idx,i in enumerate(int_time): 
        adc_param['integrate_time'] = i
        alazar_ADC = AlazarTech(adc_param)
        output = calc_1Dresonator(pin, alazar_ADC, int_time[idx])
        int_time_output.append(output)
        print(int_time[idx])
    return int_time_output

def calc_1Dresonator(Pin, Alazar_obj, int_time):
    data = None
    output = pd.DataFrame(data, columns = ['s21mag', 's21phase', 'I', 'Q', 's21mag_rms']) #initialize df

    for i in range(1,101):
        if Alazar_obj.name == 'CryoCMOS setting':
            df_data = pd.read_csv('Data\CryoRX\Pin=' + str(Pin) + 'dBm\Data (' + str(i) + ')', header = None)
        elif Alazar_obj.name == 'RT Rack setting':
            df_data = pd.read_csv('Data\Standard Setup\Pin=' + str(Pin) + 'dBm\Data (' + str(i) + ')', header = None)

        data_array = df_data.to_numpy()
        s21mag, s21phase, I, Q, s21mag_rms = Alazar_obj.post_process_data(data_array)
        new_row = {'s21mag' : s21mag, 's21phase' : s21phase, 'I' : I, 'Q' : Q, 's21mag_rms' : s21mag_rms}
        output = output.append(new_row,  ignore_index=True)
    return output

def fit_s21mag(x_val, y_val):
    x_val_midpoint = int(np.round(len(x_val)/2))
    peak = GaussianModel()
    offset = ConstantModel()
    model = peak + offset
    pars = offset.make_params(c=np.median(y_val))
    pars += peak.guess(y_val, x=x_val, amplitude=-0.05, center= x_val[x_val_midpoint])
    result = model.fit(y_val, pars, x=x_val)
    return result

def calc_SNR(int_time_output, pin, Alazar_setting, int_time):
    
    # plt.close("all")
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 2
    
    n = len(int_time_output) - 1
    col_shade = plt.cm.binary(np.linspace(0.1, 0.5, n))
    col_last = plt.cm.binary(np.linspace(1, 1, 1))
    col = np.concatenate([col_shade, col_last])
    
    vin_peak = 10 ** ((pin - 10) / 20) 
    
    SNR = []
    S = []
    N = []
    
    # DC X-axis data
    if Alazar_setting['name'] == 'CryoCMOS setting':
        v_rp_cmos  = pd.read_csv(
            'Scripts and PPT Summary/CryoRX/2020-06-22/18-15-29_qtt_scan1D/RP1.dat',
            skiprows=[0, 2], delimiter='\t')
        v_rp = v_rp_cmos['# "RP1"'].to_numpy()
    elif Alazar_setting['name'] == 'RT Rack setting':
        v_rp_RT = pd.read_csv(
            'Scripts and PPT Summary/RT Setup/2020-06-23/09-28-33_qtt_scan1D/RP1.dat',
            skiprows=[0, 2], delimiter='\t')
        v_rp = v_rp_RT['# "RP1"'].to_numpy()
    
    
    #Plotting the traces for different t_int time
    plt.figure(figsize=(5,4), dpi=150)
    for idx,output in enumerate(int_time_output):
        s21 = output["s21mag"].to_numpy() / vin_peak # convert s21mag (digitzed)
        if idx == len(int_time_output) - 1:
            plt.plot(v_rp, 20*np.log10(s21), linewidth=3, color=col[idx], label='t$_{int}$ = 1 ms \nPin=-40 dBm', zorder=10) 
        else:
            plt.plot(v_rp, 20*np.log10(s21), linewidth=2, color=col[idx]) 
            
            
        gauss_fit = fit_s21mag(v_rp[10:90], s21[10:90])
    
        S.append(abs(gauss_fit.values['height'])) # Calculate Signal Height
        # N.append(output) # Noise calculated by the RMS, first N data points
        N.append(np.std(s21[75:-1])) # Noise calculated by the RMS, first N data points
        SNR.append((S[idx] / N[idx])**2)
        
        
    plt.plot(v_rp[10:90], 20*np.log10(gauss_fit.best_fit), 'r--', linewidth=2, label='Gaussian Fit', zorder=11)
    plt.legend()
    plt.xlabel('V$_{RP} [mV]$')
    plt.ylabel('S21 [dB]') 
    plt.ylim([-30, 0])
    plt.grid(color=col[1], linestyle='--', linewidth=2)
    
    SNR_output = [int_time, SNR]
    return SNR_output

def SNR_Linear_Fitting(SNR_output):
    int_time = SNR_output[0]
    SNR = SNR_output[1] 
    t_int_linspace = np.linspace(5e-7, 5e-2, 1000)
    
    m, c = np.polyfit(np.log(int_time), np.log(SNR), 1) # fit log(y) = m*log(x) + c
    loglogfit = t_int_linspace**(m) * np.exp(c)
    t_min = np.exp(-c / m) 
    
    data = [int_time, SNR, t_int_linspace, loglogfit]
    # print(data)
    # df = pd.DataFrame[data, columns = ['Int_time', 'SNR', 't_linspace', 'loglogfit'])
    
    return data, t_min

#------------------------------------
# define settings for ADC
#------------------------------------
adc_param_CryoRX = {
    'name' : 'CryoCMOS setting', 
    'sampling_rate' : 1e9, # samples/s
    'record_length' : 0.001, # in (s)
    'channel_range' : 0.1, # in (v)
    'demodulation_frequency' : 100e6, # in (Hz)
    'integrate_time' : 0.001 # Must be less than 1ms for our case (Data only until 1ms)
    }


adc_param_RT = {
    'name' : 'RT Rack setting', 
    'sampling_rate' : 1e9, # samples/s
    'record_length' : 0.001, # in (s)
    'channel_range' : 0.04, # in (v)
    'demodulation_frequency' : 10e6, # in (Hz)
    'integrate_time' : 0.001
    }



# %%
#------------------------------------
# Main
#------------------------------------
start_time = time.time()

pin = -40
vin_peak = 10 ** ((pin - 10) / 20) 

int_time_output = []
# int_time = [8e-6, 16e-6, 32e-6, 62e-6, 124e-6, 248e-6, 496e-6, 992e-6] # Good values for SNR calculation

# int_time = np.around(np.logspace(np.log10(5e-6), np.log10(1e-3), 5), 6).tolist() # More sweep points
int_time = np.around(np.logspace(np.log10(5e-6), np.log10(1e-3), 6), 6) # More sweep points

# int_time = np.round(np.logspace(np.log10(1e-6), np.log10(1e-3), 5, base=10),6)
 
int_time_output_Cryo = calc_s21vsVrp(int_time, pin, adc_param_CryoRX)
int_time_output_RT = calc_s21vsVrp(int_time, pin, adc_param_RT)
                

# %% SNR Calculation
plt.close("all")
n = 8
col_shade = plt.cm.binary(np.linspace(0.1, 0.5, n))
col_last = plt.cm.binary(np.linspace(1, 1, 1))
col = np.concatenate([col_shade, col_last])

SNR_CryoCMOS = calc_SNR(int_time_output_Cryo, pin, adc_param_CryoRX, int_time)
SNR_RT = calc_SNR(int_time_output_RT, pin, adc_param_RT, int_time)

SNR_fit_Cryocmos, t_min_CryoCMOS = SNR_Linear_Fitting(SNR_CryoCMOS)
SNR_fit_RT, t_min_RT = SNR_Linear_Fitting(SNR_RT)

plt.figure(figsize=(8,3), dpi=150)
plt.loglog(SNR_fit_Cryocmos[2],SNR_fit_Cryocmos[3], label='CMOS IC', linewidth=3, color='b')
plt.scatter(SNR_fit_Cryocmos[0],SNR_fit_Cryocmos[1], color='b')
plt.loglog(SNR_fit_RT[2],SNR_fit_RT[3], label='RT Rack', linewidth=3, color='r')
plt.scatter(SNR_fit_RT[0],SNR_fit_RT[1], color='r')


plt.text(1e-4, 2e0, 't$_{min, CMOS}$='+str(np.round(t_min_CryoCMOS*1e6,2))+'$\mu$S', fontsize=12)
plt.text(1e-4, 2e-1, 't$_{min, Rack}$='+str(np.round(t_min_RT*1e6,2))+'$\mu$S', fontsize=12)

plt.xlabel('t$_{int}$')
plt.ylabel('SNR (a.u)') 
plt.xlim([2e-6, 2e-3])
plt.ylim([1e-1, 1e4])
plt.grid(color=col[1], linestyle='--', linewidth=2)

plt.legend()
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

# %% SNR 
#
def export_data(SNR_Data, data_name):
    int_time = SNR_Data[0].tolist()
    SNR = SNR_Data[1]
    t_int_linspace = SNR_Data[2].tolist()
    loglogfit = SNR_Data[3].tolist()
    Data_arr = [int_time, SNR, t_int_linspace, loglogfit]
    df = pd.DataFrame(Data_arr).transpose()
    df.columns = ['int_time', 'SNR', 't_int_linspace', 'LogLogFit']
    df.to_csv('Data/Processed Data/SNR Data/'+str(data_name)+'.csv', sep='\t')
    return df

df = export_data(SNR_fit_Cryocmos,'CryoCMOS_6points')
df = export_data(SNR_fit_RT,'RTRack')

