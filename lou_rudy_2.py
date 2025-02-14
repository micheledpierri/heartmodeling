import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import odeint

# Costanti fisiologiche
C_m = 1.0  # µF/cm²
E_Na = 50.0  # mV
E_K = -85.0  # mV
E_L = -60.0  # mV

# Funzioni per le variabili di gating
def alpha_m(V): return 0.32 * (V + 47.13) / (1 - np.exp(-(V + 47.13) / 5))
def beta_m(V): return 0.08 * np.exp(-V / 11)
def alpha_h(V): return 0.135 * np.exp((80 + V) / -6.8)
def beta_h(V): return 3.56 * np.exp(0.079 * V) + 310000 * np.exp(0.35 * V)
def alpha_n(V): return 0.02 * (V + 50) / (1 - np.exp(-(V + 50) / 10))
def beta_n(V): return 0.5 * np.exp(-(V + 55) / 40)

# Equazioni Luo-Rudy
def luo_rudy_model(y, t, g_Na, g_K, g_L, I_ext):
    V, m, h, n = y
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)

    dVdt = (I_ext - (I_Na + I_K + I_L)) / C_m
    return [dVdt, dmdt, dhdt, dndt]

st.title("Simulazione del modello di Luo-Rudy (1991)")

# Input interattivi
g_Na = st.slider("Conduttanza \( g_{Na} \)", 0.0, 150.0, 120.0)
g_K = st.slider("Conduttanza \( g_K \)", 0.0, 50.0, 36.0)
g_L = st.slider("Conduttanza \( g_L \)", 0.0, 1.0, 0.3)
I_ext = st.slider("Corrente \( I_{ext} \)", -10.0, 20.0, 5.0)

# Simulazione
t = np.linspace(0, 300, 5000)
sol = odeint(luo_rudy_model, [-75, 0.02, 0.8, 0.1], t, args=(g_Na, g_K, g_L, I_ext))

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=sol[:, 0], mode="lines", name="Potenziale d'azione"))
fig.update_layout(title="Luo-Rudy (1991)", xaxis_title="Tempo (ms)", yaxis_title="Potenziale (mV)")
st.plotly_chart(fig)
