import numpy as np 
import matplotlib.pyplot as plt


N = 1000           # Total number of simulation time points (1000 steps = 100 ms)
g_L = 0.7          # Membrane leak conductance [nS]
c_m = 20           # Membrane capacitance [nF/mm^2]
tau_m = c_m / g_L  # Membrane time constant (~28.57 ms), governs voltage decay speed
v_reset = -65      # Reset membrane potential after a spike [mV]
v_thresh = -50     # Voltage threshold required to fire an action potential [mV]
dt = 0.1           # Integration time step size [ms]
E_L = -75          # Resting / leak reversal potential [mV]


v = np.zeros(N)    # Array storing membrane potential trajectory over time
v[0] = v_reset     # Initialize membrane voltage to reset potential


N_input = 5                 # Number of incoming presynaptic input channels
tau_plus = 20.0             # LTP (Long-Term Potentiation) time decay window [ms]
tau_minus = 20.0            # LTD (Long-Term Depression) time decay window [ms]
A_plus = 1.5                # STDP potentiation magnitude (weight enhancement)
A_minus = 1.55              # STDP depression magnitude (weight suppression)
w_max = 25.0                # Upper bound limit for any individual synaptic weight


tau_homeo = 100.0   # Time constant for tracking average postsynaptic firing rate [ms]
R_target = 0.06     # Desired target firing rate (0.06 spikes/ms = 60 Hz)
eta_homeo = 0.0005  # Learning rate multiplier for homeostatic synaptic scaling
R_track = 0.0       # Integrated running estimate of postsynaptic spiking activity



w = np.ones(N_input) * 5.0              # Initialize all 5 synaptic weights to baseline 5.0
P = np.zeros(N_input)                   # Presynaptic traces (detect pre-before-post timing)
M = 0.0                                 # Postsynaptic trace (detects post-before-pre timing)
I_syn = 0.0                             # Aggregated synaptic input current
tau_syn = 5.0                           # Synaptic current exponential decay time constant [ms]

input_spikes = np.zeros((N_input, N), dtype=bool)  # Grid tracking input spike events
w_history = np.zeros((N_input, N))                 # History matrix for plotting weight evolution
R_history = np.zeros(N)                            # History array for tracking estimated firing rate



# Inject structured temporal correlations across specific channels (0, 2, 4) every 6 ms
for i in range(0, N, 60):
    if i + 20 < N:
        input_spikes[0, i] = True         # Base spike at channel 0
        input_spikes[2, i + 5] = True     # Channel 2 fires 0.5 ms after channel 0
        input_spikes[4, i + 12] = True    # Channel 4 fires 1.2 ms after channel 0

# Add random Poisson-like background noise (0.2% probability per time step across all channels)
noise = np.random.rand(N_input, N) < 0.002
input_spikes = input_spikes | noise       # Merge structured input patterns with noise spikes



for i in range(N - 1):
    # Store current synaptic weights and firing rate estimate for plotting
    w_history[:, i] = w.copy()
    R_history[i] = R_track

    # Continuous exponential decay of variables toward baseline/zero
    I_syn -= (I_syn / tau_syn) * dt
    P -= (P / tau_plus) * dt
    M -= (M / tau_minus) * dt
    R_track -= (R_track / tau_homeo) * dt

    #presynaptic spike processing 
    for j in range(N_input):
        if input_spikes[j, i]:
            I_syn += w[j]                                # Add current weight to total synaptic input
            P[j] += 1.0                                  # Increment presynaptic STDP trace
            w[j] = np.clip(w[j] - A_minus * M, 0, w_max) # LTD: weaken weight if postsynaptic neuron fired recently

    #membrane potential integration  
    dv = (-(v[i] - E_L) + (2.50 * I_syn) / g_L) * dt / tau_m
    v_new = v[i] + dv

    #postsynaptic spike detection & update
    if v_new >= v_thresh:
        v[i + 1] = v_reset                               # Reset membrane voltage after spike
        M += 1.0                                         # Increment postsynaptic STDP trace
        w = np.clip(w + A_plus * P, 0, w_max)           # LTP: strengthen weights of recently active inputs
        R_track += 1.0 / tau_homeo                      # Increment estimated firing rate tracker
    else:
        v[i + 1] = v_new                                 # Update membrane voltage without spike

    #Homeostatic synaptic scaling  
    # Multiplicatively scale weights up/down to keep firing rate (R_track) close to target (R_target)
    dw_scaling = eta_homeo * (R_target - R_track) * w * dt
    w = np.clip(w + dw_scaling, 0, w_max)

# Save final step state to history arrays
w_history[:, -1] = w.copy()
R_history[-1] = R_track

print("Simulation complete!")
print(f"Final synaptic weights across 5 channels: {np.round(w, 2)}")


time = np.arange(N) * dt
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

#1st graph: membrane voltage trajectory
axs[0].plot(time, v, color='black', lw=1)
axs[0].axhline(v_thresh, color='red', linestyle='--', label='Threshold (-50 mV)')
axs[0].set_ylabel('v (mV)')
axs[0].set_title('Membrane Potential Dynamics with STDP and Synaptic Scaling')
axs[0].legend(loc='upper right')
axs[0].grid(True, alpha=0.3)

#2nd graph: synaptic weight evolution
for j in range(N_input):
    style = '-' if j in [0, 2, 4] else '--'
    lw = 2 if j in [0, 2, 4] else 1
    axs[1].plot(time, w_history[j, :], label=f'Channel {j}', linestyle=style, linewidth=lw)

axs[1].set_ylabel('Synaptic Weight w')
axs[1].set_title('Synaptic Weight Dynamics (Correlated Channels: 0, 2, 4)')
axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
axs[1].grid(True, alpha=0.3)

#3rd graph: estimated firing rate tracking vs. target rate
axs[2].plot(time, R_history, color='purple', label='Estimated Rate (R_track)')
axs[2].axhline(R_target, color='green', linestyle='--', label=f'Target Rate (R_target = {R_target})')
axs[2].set_xlabel('Time (ms)')
axs[2].set_ylabel('Firing Rate')
axs[2].set_title('Homeostatic Plasticity (Synaptic Scaling Tracking)')
axs[2].legend(loc='upper right')
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()