# Computational-Neuronal-Models-


This repository displays some neuronal models written in Python code, calculated both with Euler's and fourth-order Runge-Kutta's method. Some bibliographic references are added in order to make the viewer more comfortable with the mathematical framework of each model.

## Key Highlights
* **Numerical Methods:** Implementation of 1st-order Euler and 4th-order Runge-Kutta (RK4) integration schemes for differential equations.
* **Biophysical Realism:** Modeling ion channel gating dynamics, membrane capacitance, and action potential propagation.

---

## Simulated Models

### 1. Leaky Integrate-and-Fire model (LIF)
A simple neuron that fires when a certain threshold is reached, and then goes back to its resting voltage value.
* **Reference:** [Neuronal Dynamics - Chapter 1](https://neuronaldynamics.epfl.ch/online/Ch1.html)

### 2. Hodgkin-Huxley model (HH)
A neuron where its ionic currents, which affect membrane voltage change, are linked to ionic voltage-dependent gates.
* **Reference:** [Neuronal Dynamics - Chapter 2](https://neuronaldynamics.epfl.ch/online/Ch2.html)

### 3. Spike-Timing-Dependent Plasticity (STDP) & Synaptic Scaling
A little project concerning plasticity is also included: a simple neuron linked to five synaptic inputs that modify their weights according to synaptic scaling mechanisms.
* **References:**
  * [Scholarpedia - Spike-timing dependent plasticity](http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity) (STDP foundational model)
  * [Journal of Neuroscience - Synaptic Scaling](https://www.jneurosci.org/content/18/24/10464) (Synaptic scaling mathematical model)

> *Note:* Sometimes parameters were chosen under different conventions (compared to the references) and fine-tuned in order to improve the output.

---

## Some Visual Outputs

| Leaky Integrate-and-Fire (LIF) | Hodgkin-Huxley (HH) | STDP & Synaptic Scaling |
| :---: | :---: | :---: |
| ![LIF Model Plot](Computational%20Neuronal%20Models/Basic%20Plots/LIF_plot.png) | ![HH Model Plot](Computational%20Neuronal%20Models/Basic%20Plots/HH_plot.png) | ![STDP Plot](Computational%20Neuronal%20Models/Basic%20Plots/STPD_plot.png) |


## Requirements & Execution

To run the scripts, make sure you have the following Python packages installed:

```bash
pip install numpy matplotlib scipy
