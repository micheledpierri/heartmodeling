import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
C_m = 1.0  # µF/cm²
E_Na = 50.0  # mV
E_K = -85.0  # mV
E_L = -60.0  # mV
E_Ca = 120.0  # mV

# Gating functions (added missing α_d and β_d)
def alpha_m(V): return 0.32 * (V + 47.13) / (1 - np.exp(-(V + 47.13) / 5))
def beta_m(V): return 0.08 * np.exp(-V / 11)
def alpha_h(V): return 0.135 * np.exp((80 + V) / -6.8)
def beta_h(V): return 3.56 * np.exp(0.079 * V) + 310000 * np.exp(0.35 * V)
def alpha_n(V): return 0.02 * (V + 50) / (1 - np.exp(-(V + 50) / 10))
def beta_n(V): return 0.5 * np.exp(-(V + 55) / 40)

# Calcium activation (corrected)
def alpha_d(V): return 0.095 * np.exp(-(V + 50) / 10) / (1 + np.exp(-(V + 50) / 10))
def beta_d(V): return 0.07 * np.exp(-(V + 70) / 20) / (1 + np.exp(-(V + 70) / 20))

# Luo-Rudy Model with \( I_{CaL} \)
def luo_rudy_model(y, t, g_Na, g_K, g_L, g_CaL, I_ext):
    V, m, h, n, d = y
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dddt = alpha_d(V) * (1 - d) - beta_d(V) * d

    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)
    I_CaL = g_CaL * d * (V - E_Ca)

    dVdt = (I_ext - (I_Na + I_K + I_L + I_CaL)) / C_m
    return [dVdt, dmdt, dhdt, dndt, dddt]

# Streamlit Interface
st.title("Luo-Rudy Model (1991) with \( I_{CaL} \)")

# Interactive sliders
g_Na = st.slider("Sodium Conductance (g_Na)", 0.0, 150.0, 120.0)
g_K = st.slider("Potassium Conductance (g_K)", 0.0, 50.0, 36.0)
g_L = st.slider("Leak Conductance (g_L)", 0.0, 1.0, 0.3)
g_CaL = st.slider("Calcium Conductance (g_CaL)", 0.0, 1.0, 0.09)
I_ext = st.slider("External Current (I_ext)", -10.0, 20.0, 5.0)

# Solve ODE
t = np.linspace(0, 300, 5000)
sol = odeint(luo_rudy_model, [-75, 0.02, 0.8, 0.1, 0.01], t, args=(g_Na, g_K, g_L, g_CaL, I_ext))

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, sol[:, 0], 'b', label="Membrane Potential (mV)")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Potential (mV)")
ax.set_title("Luo-Rudy Model (1991) with \( I_{CaL} \)")
ax.legend()
ax.grid()

st.pyplot(fig)
