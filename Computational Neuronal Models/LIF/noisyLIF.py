#Add randomness to I(t) and v_new
import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal
N = 1000           # number of measurememnts (time points)
g_L = 0.7          # leak conductance [nS]
c_m = 20           # conductance [nF/mm^2]
tau_m = c_m/g_L    # membrane time constant [ms]
I = np.random.normal(100, 25)             # randomized input current [nA]
v_reset = -65      # reset potential [mV]
v_thresh = -50     # threshold [mV]
dt = 0.1           # time step
E_L = -75          # leak reversal potential [mV]
sigma= 1.0
v = np.zeros(N)     # list of potential values, start at v_reset
v[0] = v_reset
for i in range(N - 1): # loop over all time points
    # increment of membrane potential
    dv = (-(v[i]-E_L) +I/g_L)* dt /tau_m
    noise = np.random.normal(0, np.sqrt(dt))
    v_new = v[i] + dv*dt + sigma*noise    # new membrane potential value
    if v_new >= v_thresh:  # check if new value exceeds threshold
        v[i+1] = v_reset# add reset value, if true
    else:
        v[i+1] = v_new  # add new value to the list otherwise
#v = np.asarray(v)
fs = 1000
t = np.linspace(0, 2, 2 * fs, endpoint=False)

frequencies, times, spectrogram = signal.spectrogram(v, fs=fs, nperseg=256)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor="white")

#1st graph: voltage over time 
ax1.plot(v, color='black', lw=4)
ax1.axhline(v_thresh, linestyle='dashed', lw=1, label='Threshold', color='red')
ax1.axhline(v_reset, linestyle='dashed', lw=1, label='Reset', color='blue')
ax1.legend()
ax1.set_xlabel('Time point')
ax1.set_ylabel('V, [mV]')
ax1.set_title('Membrane potential')

#2nd graph: spectrogram
pcm = ax2.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='gouraud', cmap='viridis')
ax2.set_title('Time-Frequency plot (Spectrogram)')
ax2.set_ylabel('frequency [Hz]')
ax2.set_xlabel('time [sec]')
cbar = fig.colorbar(pcm, ax=ax2)
cbar.set_label('intensity [dB]')
fig.tight_layout()
plt.show()

