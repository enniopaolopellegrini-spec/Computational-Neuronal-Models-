import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
N = 1000           # number of measurememnts (time points)
g_L = 0.7          # leak conductance [nS]
r_m = 1/g_L        # resistance [nΩ]
c_m = 20           # conductance [nF/mm^2]
tau_m = c_m/g_L        # membrane time constant [ms]
I = np.random.normal(100,25)             # input current [nA]
v_reset = -65      # reset potential [mV]
v_thresh = -50     # threshold [mV]
dt = 0.1           # time step
E_L = -75          # leak reversal potential [mV]
sigma = 1.0


t_rk = np.linspace(0, N*dt, N+1)
v_rk = np.zeros(N + 1)
v_rk[0] = v_reset

def f(V_val, t_val):
    return (E_L - V_val + r_m*I)/tau_m

for i in range(N):
    t_n = t_rk[i]
    v_n = v_rk[i]

    k1 = f(v_n, t_n)
    k2 = f(v_n + (dt/2.0)*k1, t_n + dt/2.0)
    k3 = f(v_n + (dt/2.0)*k2, t_n + dt/2.0)
    k4 = f(v_n + dt*k3, t_n + dt)
    noise = np.random.normal(0, np.sqrt(dt))

    v_new = v_n + (k1 + 2*k2 + 2*k3 + k4)*(dt/6.0) + sigma*noise
    if v_new >= v_thresh:  # check if new value exceeds threshold
        v_rk[i+1] = v_reset
    else:
        v_rk[i+1] = v_new


fs = 1000
frequencies, times, spectrogram = signal.spectrogram(v_rk, fs=fs, nperseg=256)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), facecolor="white")
##1st graph: voltage over time 
ax1.plot(v_rk, color='black', lw=4)
ax1.axhline(v_thresh, linestyle='dashed', lw=1, label='Threshold', color='red')
ax1.axhline(v_reset, linestyle='dashed', lw=1, label='Reset', color='blue')
ax1.legend()
ax1.set_xlabel('Time point')
ax1.set_ylabel('V, [mV]')
ax1.set_title('Membrane potential', fontsize=14, fontweight='bold')

#2nd graph: spectrogram 
pcm = ax2.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='gouraud', cmap='viridis')
ax2.set_title('Time-Frequency plot (Spectrogram)')
ax2.set_ylabel('frequency [Hz]')
ax2.set_xlabel('time [sec]')
cbar = fig.colorbar(pcm, ax=ax2)
cbar.set_label('intensity [dB]')
fig.tight_layout()
plt.show()
