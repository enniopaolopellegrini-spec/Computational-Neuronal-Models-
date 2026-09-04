import numpy as np
import matplotlib.pyplot as plt


N = 5000       # number of step 
dt = 0.01      # time step [ms]

# Variabili di tempo richieste
time_ms = np.arange(N) * dt
time_sec = time_ms / 1000.0

# --- Parametri Hodgkin-Huxley ---
c_m  = 1.0     # Membrane conductance [uF/cm^2]
g_Na = 120.0   # Na conductance [mS/cm^2]
g_K  = 36.0    # K conductance [mS/cm^2]
g_L  = 0.3     # leak conductance [mS/cm^2]

E_Na = 50.0    # Na inversion potential [mV]
E_K  = -77.0   # K  inversion potential [mV]
E_L  = -54.387 # Leak inversion potential [mV]

# Input current (costant between 10ms and 40ms)
I_ext = np.where((time_ms >= 10.0) & (time_ms <= 40.0), 10.0, 0.0)

# α and β functions for each gating variable 
def alpha_m(v): return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
def beta_m(v):  return 4.0 * np.exp(-(v + 65.0) / 18.0)

def alpha_h(v): return 0.07 * np.exp(-(v + 65.0) / 20.0)
def beta_h(v):  return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))

def alpha_n(v): return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
def beta_n(v):  return 0.125 * np.exp(-(v + 65.0) / 80.0)

# System of differential equations [dv/dt, dm/dt, dh/dt, dn/dt] 
def hh_derivatives(v, m, h, n, I):
    I_Na = g_Na * (m**3) * h * (v - E_Na)
    I_K  = g_K  * (n**4) * (v - E_K)
    I_L  = g_L  * (v - E_L)
    I_ion = I_Na + I_K + I_L
    
    dv_dt = (I - I_ion) / c_m
    dm_dt = alpha_m(v) * (1.0 - m) - beta_m(v) * m
    dh_dt = alpha_h(v) * (1.0 - h) - beta_h(v) * h
    dn_dt = alpha_n(v) * (1.0 - n) - beta_n(v) * n
    
    return np.array([dv_dt, dm_dt, dh_dt, dn_dt])

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

# Runge-Kutta 4th order integration method (RK4) 
for i in range(N - 1):
    I_curr = I_ext[i]
    state = np.array([v[i], m[i], h[i], n[i]])
    
    # K1
    k1 = hh_derivatives(state[0], state[1], state[2], state[3], I_curr)
    
    # K2
    s_k2 = state + 0.5 * dt * k1
    k2 = hh_derivatives(s_k2[0], s_k2[1], s_k2[2], s_k2[3], I_curr)
    
    # K3
    s_k3 = state + 0.5 * dt * k2
    k3 = hh_derivatives(s_k3[0], s_k3[1], s_k3[2], s_k3[3], I_curr)
    
    # K4
    s_k4 = state + dt * k3
    k4 = hh_derivatives(s_k4[0], s_k4[1], s_k4[2], s_k4[3], I_curr)
    
    # State update 
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    v[i+1], m[i+1], h[i+1], n[i+1] = next_state

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