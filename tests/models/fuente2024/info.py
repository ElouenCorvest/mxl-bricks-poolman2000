nPSII = 1
nPSI = 1
PQtot = 7
Atot = 1000
k1p = 25000
k1m = 2500
k2p = 100
k2m = 10
k3 = 0.05
k4 = 0.004
k5 = 100
k6 = 10
k7 = 500
kX = 1
sigmaI = 1
L12 = 10000
FQmax = 0.6
n_npq = 5.3
bH = 0.01
NA = 6.02214076 * 10**17
volLum = 2.66 * 10**(-21)
volStr = 2.09 * 10**(-20)
cEqP = 4.30*10**(-8)
Hstroma = 10**(-7.8)/10**(-6)
kQ = 10**(-6)/10**(-6)
ChlF0 = 0.25

import numpy as np
import pandas as pd
from model import get_fuente2024
import mxlbricks.names as n

u0 = get_fuente2024().get_parameter_values()[n.pfd()]
u1 = get_fuente2024().get_parameter_values()[n.pfd("_add")]
T = 1 / get_fuente2024().get_parameter_values()["f"]
Pi = np.pi
t = 0

y0 = get_fuente2024().get_initial_conditions()

FQact=y0["Q_active"]
PQ = y0["Plastoquinone (oxidised)"]
PIox = y0["PSI_ox"]
HL = y0["protons_lumen"]
ATP = y0["ATP_stroma"]

Light = u0 + u1*np.cos(2 * Pi * t / T)
sigmaII = (1 - FQmax*FQact)
RCIIclosed = 1/(1 + k1p*PQ/(sigmaII*Light + k1m*(PQtot - PQ)))
# print(f"{k1p=}, {PQ=}, {sigmaII=}, {Light=}, {k1m=}, {PQtot=}")
RCIIopen = k1p*PQ/((sigmaII + k1m*(PQtot - PQ)) + k1p*PQ)


vPSII = nPSII*sigmaII*Light*(1 - RCIIclosed)
vPSI = nPSI*sigmaI*L12*Light/(L12 + Light)*(nPSI - PIox)
# print(f"{nPSI=}, {sigmaI=}, {L12=}, {Light=}, {PIox=}, {vPSI=}")
v1 = k1p*RCIIclosed*PQ - k1m*(RCIIopen)*(PQtot - PQ)

v2 = k2p*(PQtot - PQ)*PIox - k2m*PQ*(nPSI - PIox)
v3 = k3*(1 - FQact)/(1 + (kQ/HL)**n_npq)
v4 = k4*FQact
v5 = k5*((Atot - ATP) - ATP*(Hstroma/HL)**(14/3)/cEqP)
v6 = k6*ATP
v7 = k7*(HL - Hstroma)
vX = kX*(PQtot - PQ)

PQ_dt = (v2 - v1)/2 + vX
HL_dt = bH*((vPSII + v2 )/(NA*volLum) - (14/3)*(volStr/volLum)*v5) - v7
FQact_dt = v3 - v4
ATP_dt = v5 - v6
PIox_dt = vPSI - v2

def fluxes():
    here = pd.Series({
        "v_PSII": vPSII,
        "v_PSI": vPSI,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "v4": v4,
        "v5": v5,
        "v6": v6,
        "v7": v7,
        "v_X": vX
    })
    
    diff = get_fuente2024().get_fluxes() - here
    diff.name = "flux_diff"
    
    return diff[diff != 0]

def rhs():
    here = pd.Series({
        "Plastoquinone (oxidised)": PQ_dt,
        "protons_lumen": HL_dt,
        "Q_active": FQact_dt,
        "ATP_stroma": ATP_dt,
        "PSI_ox": PIox_dt
    })
    
    diff = get_fuente2024().get_right_hand_side() - here
    diff.name = "rhs_diff"
    
    return diff[diff != 0]

m = get_fuente2024()
m.update_parameters({n.pfd(): 500})
print(m.get_initial_conditions())
# print(f"INfo: {k1p=}, {RCIIclosed=}, {PQ=}, {k1m=}, {RCIIopen=}, {PQtot=}, {v1=}")