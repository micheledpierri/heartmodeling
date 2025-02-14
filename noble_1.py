import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
C_m = 1.2
E_Na = 40.0
E_K = -100.0
E_L = -60.0
E_Ca = 120.0

# Gating functions
def alpha_d(V): return 0.095 * np.exp(-(V + 50) / 10) / (1 + np.exp(-(V + 50) / 10))
def beta_d(V): return 0.07 * np.exp(-(V + 70) / 20) / (1 + np.exp(-(V + 70) / 20))

# Noble Model with Calcium Current
def noble_model(y, t, g_Na, g_K, g_L, g_CaL, I_ext):
    V, m, h, n, d = y
    dmdt = alpha_d(V) * (1 - m) - beta_d(V) * m
    dhdt = alpha_d(V) * (1 - h) - beta_d(V) * h
    dndt = alpha_d(V) * (1 - n) - beta_d(V) * n
    dddt = alpha_d(V) * (1 - d) - beta_d(V) * d

    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)
    I_CaL = g_CaL * d * (V - E_Ca)

    dVdt = (I_ext - (I_Na + I_K + I_L + I_CaL)) / C_m
    return [dVdt, dmdt, dhdt, dndt, dddt]

# Streamlit Interface
st.title("Noble Model (1962) with \( I_{CaL} \)")

# Interactive sliders
g_Na = st.slider("Sodium Conductance (g_Na)", 0.0, 100.0, 70.0)
g_K = st.slider("Potassium Conductance (g_K)", 0.0, 50.0, 30.0)
g_L = st.slider("Leak Conductance (g_L)", 0.0, 1.0, 0.1)
g_CaL = st.slider("Calcium Conductance (g_CaL)", 0.0, 1.0, 0.09)
I_ext = st.slider("External Current (I_ext)", -10.0, 20.0, 5.0)

# Solve ODE
t = np.linspace(0, 100, 2000)
sol = odeint(noble_model, [-75, 0.02, 0.8, 0.1, 0.01], t, args=(g_Na, g_K, g_L, g_CaL, I_ext))

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, sol[:, 0], 'b', label="Membrane Potential (mV)")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Potential (mV)")
ax.set_title("Noble Model (1962) with \( I_{CaL} \)")
ax.legend()
ax.grid()

st.pyplot(fig)
