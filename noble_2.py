import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import odeint

# Costanti fisiologiche del modello di Noble
C_m = 1.2  # µF/cm² (capacitanza)
E_Na = 40.0  # mV (potenziale equilibrio Na+)
E_K = -100.0  # mV (potenziale equilibrio K+)
E_L = -60.0  # mV (potenziale equilibrio Leak)

# Funzioni per le variabili di gating
def alpha_m(V): return 0.1 * (V + 48) / (1 - np.exp(-(V + 48) / 15))
def beta_m(V): return 0.12 * (V + 8) / (np.exp((V + 8) / 5) - 1)
def alpha_h(V): return 0.17 * np.exp(-0.08 * (V + 57))
def beta_h(V): return 1 / (1 + np.exp(-0.15 * (V + 23)))
def alpha_n(V): return 0.0001 * (V + 50) / (1 - np.exp(-(V + 50) / 10))
def beta_n(V): return 0.002 * np.exp(-0.05 * (V + 55))

# Equazioni differenziali del modello di Noble
def noble_model(y, t, g_Na, g_K, g_L, I_ext):
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
st.title("Simulazione del modello di Noble (1962)")

# Input interattivi per la modifica dei parametri
g_Na = st.slider("Conduttanza \( g_{Na} \) (mS/cm²)", 0.0, 100.0, 70.0)
g_K = st.slider("Conduttanza \( g_K \) (mS/cm²)", 0.0, 50.0, 30.0)
g_L = st.slider("Conduttanza \( g_L \) (mS/cm²)", 0.0, 1.0, 0.1)
I_ext = st.slider("Corrente esterna \( I_{ext} \) (µA/cm²)", -10.0, 20.0, 5.0)

# Condizioni iniziali
V0, m0, h0, n0 = -75, 0.02, 0.8, 0.1
y0 = [V0, m0, h0, n0]

# Tempo di simulazione
t = np.linspace(0, 100, 2000)  # Simulazione più lunga rispetto a HH (100 ms)

# Risoluzione equazioni differenziali
sol = odeint(noble_model, y0, t, args=(g_Na, g_K, g_L, I_ext))

# Creazione del grafico interattivo con Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=sol[:, 0], mode="lines", name="Potenziale di membrana (mV)"))

fig.update_layout(
    title="Potenziale d'azione - Modello di Noble (1962)",
    xaxis_title="Tempo (ms)",
    yaxis_title="Potenziale (mV)",
    template="plotly_dark",
    hovermode="x",
)

st.plotly_chart(fig)
