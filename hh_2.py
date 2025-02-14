import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import odeint

# Definizione delle costanti fisiologiche
C_m = 1.0  # µF/cm²
E_Na = 50.0  # mV
E_K = -77.0  # mV
E_L = -54.4  # mV

# Funzioni per le variabili di gating
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V): return 4.0 * np.exp(-0.0556 * (V + 65))
def alpha_h(V): return 0.07 * np.exp(-0.05 * (V + 65))
def beta_h(V): return 1 / (1 + np.exp(-0.1 * (V + 35)))
def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V): return 0.125 * np.exp(-0.0125 * (V + 65))

# Equazioni differenziali per Hodgkin-Huxley
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

# Interfaccia Streamlit
st.title("Simulazione del modello di Hodgkin-Huxley con Plotly")

# Input interattivi
g_Na = st.slider("Conduttanza \( g_{Na} \) (mS/cm²)", 0.0, 150.0, 120.0)
g_K = st.slider("Conduttanza \( g_K \) (mS/cm²)", 0.0, 50.0, 36.0)
g_L = st.slider("Conduttanza \( g_L \) (mS/cm²)", 0.0, 1.0, 0.3)
I_ext = st.slider("Corrente esterna \( I_{ext} \) (µA/cm²)", 0.0, 20.0, 10.0)

# Condizioni iniziali
V0, m0, h0, n0 = -65, 0.05, 0.6, 0.32
y0 = [V0, m0, h0, n0]

# Tempo di simulazione
t = np.linspace(0, 50, 1000)  # 50 ms

# Risoluzione equazioni differenziali
sol = odeint(hodgkin_huxley, y0, t, args=(g_Na, g_K, g_L, I_ext))

# Creazione del grafico interattivo con Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=sol[:, 0], mode="lines", name="Potenziale di membrana (mV)"))

fig.update_layout(
    title="Potenziale d'azione - Modello Hodgkin-Huxley",
    xaxis_title="Tempo (ms)",
    yaxis_title="Potenziale (mV)",
    template="plotly_dark",
    hovermode="x",
)

st.plotly_chart(fig)
