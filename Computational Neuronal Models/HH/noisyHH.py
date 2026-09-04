import numpy as np
import matplotlib.pyplot as plt


N = 5000       # number of steps
dt = 0.01      # timestep [ms]
time_ms = np.arange(N) * dt
sigma = 7.0    # standard deviation for noise
tau_v = 5.0


c_m  = 1.0     # Membrane capacity [uF/cm^2]
g_Na = 120.0   # Na conductance [mS/cm^2]
g_K  = 36.0    # K  conductance [mS/cm^2]
g_L  = 0.3     # leak conductance [mS/cm^2]

E_Na = 50.0    # Na inversion potential [mV]
E_K  = -77.0   # K  inversion potential [mV]
E_L  = -54.387 # Leak inversion potential [mV]


#Model input current as a randomized current with an Ornstein-Uhlenbeck process, where:
# dI(t) = ((I_mean - I(t)) / tau_v) * dt + sigma * sqrt(2 / t) * dW(t)
I_mean = 0.0 
I_noise = np.zeros(N)
I_noise[0] = I_mean

noise = sigma * np.sqrt(2.0 / tau_v) * np.sqrt(dt)
xi = np.random.normal(0, 1, N)

for i in range(N-1):
    drift = -((I_noise[i] - I_mean) / tau_v) * dt
    diffusion = noise * xi[i]
    I_noise[i+1] = I_noise[i] + drift + diffusion

#  α and β functions for each gating variable 
def alpha_m(v): return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
def beta_m(v):  return 4.0 * np.exp(-(v + 65.0) / 18.0)

def alpha_h(v): return 0.07 * np.exp(-(v + 65.0) / 20.0)
def beta_h(v):  return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))

def alpha_n(v): return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
def beta_n(v):  return 0.125 * np.exp(-(v + 65.0) / 80.0)

# Setting the voltage and gating variable arrays 
v = np.zeros(N)
m = np.zeros(N)
n = np.zeros(N)
h = np.zeros(N)

# Initial resting conditions (V_0 = -65 mV)
v[0] = -65.0
m[0] = alpha_m(v[0]) / (alpha_m(v[0]) + beta_m(v[0]))
h[0] = alpha_h(v[0]) / (alpha_h(v[0]) + beta_h(v[0]))
n[0] = alpha_n(v[0]) / (alpha_n(v[0]) + beta_n(v[0]))

# Euler's integration method 
for i in range(N - 1):
    v_i = v[i]
    
    # Currents 
    I_Na = g_Na * (m[i]**3) * h[i] * (v_i - E_Na)
    I_K  = g_K  * (n[i]**4) * (v_i - E_K)
    I_L  = g_L  * (v_i - E_L)
    I_ion = I_Na + I_K + I_L
    
    # V update 
    dv = (I_noise[i] - I_ion) / c_m
    v[i+1] = v_i + dv * dt
    
    #gating update 
    dm = alpha_m(v_i) * (1.0 - m[i]) - beta_m(v_i) * m[i]
    dh = alpha_h(v_i) * (1.0 - h[i]) - beta_h(v_i) * h[i]
    dn = alpha_n(v_i) * (1.0 - n[i]) - beta_n(v_i) * n[i]
    
    m[i+1] = m[i] + dm * dt
    h[i+1] = h[i] + dh * dt
    n[i+1] = n[i] + dn * dt

# Storing the potassium and sodium currents as arrays
I_Na_vec = g_Na * (m**3) * h * (v - E_Na)
I_K_vec  = g_K  * (n**4) * (v - E_K)



fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

#1st graph: Membrane potential
axs[0].plot(time_ms, v, color='black', lw=1.5)
axs[0].set_ylabel('V [mV]')
axs[0].set_title('Simulazione Hodgkin-Huxley')

#2nd graph: gating variables 
axs[1].plot(time_ms, m, label='m (Na+ att.)', color='red')
axs[1].plot(time_ms, h, label='h (Na+ inatt.)', color='orange')
axs[1].plot(time_ms, n, label='n (K+ att.)', color='blue')
axs[1].set_ylabel('Gating')
axs[1].legend(loc='upper right')

#3rd graph: Ionic currents 
axs[2].plot(time_ms, I_Na_vec, label='I_Na', color='red')
axs[2].plot(time_ms, I_K_vec, label='I_K', color='blue')
axs[2].set_ylabel('Corrente [uA/cm^2]')
axs[2].set_xlabel('Tempo [sec]')
axs[2].legend(loc='upper right')

plt.tight_layout()
plt.show()