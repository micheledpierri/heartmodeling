import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import odeint

# Constants
C_m = 1.0
E_Na = 50.0
E_K = -77.0
E_L = -54.4

# Gating functions
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V): return 4.0 * np.exp(-0.0556 * (V + 65))
def alpha_h(V): return 0.07 * np.exp(-0.05 * (V + 65))
def beta_h(V): return 1 / (1 + np.exp(-0.1 * (V + 35)))
def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V): return 0.125 * np.exp(-0.0125 * (V + 65))

# Hodgkin-Huxley Model
def hodgkin_huxley(y, t, g_Na, g_K, g_L, I_ext):
    V, m, h, n = y
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)

    dVdt = (I_ext - (I_Na + I_K + I_L)) / C_m
    return [dVdt, dmdt, dhdt, dndt]

# Streamlit Interface
st.title("Hodgkin-Huxley Model (1952)")

g_Na = st.slider("Sodium Conductance (g_Na)", 0.0, 150.0, 120.0)
g_K = st.slider("Potassium Conductance (g_K)", 0.0, 50.0, 36.0)
g_L = st.slider("Leak Conductance (g_L)", 0.0, 1.0, 0.3)
I_ext = st.slider("External Current (I_ext)", 0.0, 20.0, 10.0)

# Solve ODE
t = np.linspace(0, 50, 1000)
sol = odeint(hodgkin_huxley, [-65, 0.05, 0.6, 0.32], t, args=(g_Na, g_K, g_L, I_ext))

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, sol[:, 0], 'b', label="Membrane Potential (mV)")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Potential (mV)")
ax.set_title("Hodgkin-Huxley Model (1952)")
ax.legend()
ax.grid()

st.pyplot(fig)
