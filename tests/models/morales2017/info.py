import numpy as np

# Vars
PGA = 50 # in umol/m^2
RuBP = 50 # in umol/m^2
fRB = 0.25
fP = 0
fZ = 0
#state PhiqE = 0
alphar = 1
PSIId = 0
fR = 0
PR = 0 # in umol/m^2
Cc = 380 # in umol/mol
Ccyt = 380 # in umol/mol
Ci = 380 # in umol/mol
Ca = 380 # in umol/mol
H2OS = 20 # in mmol/mol
gsw = 0.09 # in mol/m^2/s
#state Tl = 298 in K

# Params
RB = 15.90 # in umol/m^2
Vch = 10 # in mL/m^2
KmRuBP = 0.02 # in mM
KiPGA = 0.84 # in mM
Kc25 = 4.16 # in 1/s
Tl = 298 # in K
Tref = 298.15 # in K
DHaKc = 41.82 # in kJ/mol
R = 8.31 # in J/mol/K
Kmc25 = 261.7 # in umol/mol
DHaKmc = 49.43 # in kJ/mol
O2 = 210 # in mmol/mol
Kmo25 = 198.5 # in mmol/mol
DHaKmo = 29.08 # in kJ/mol
Ko25 = 1.26 # in 1/s
DHaKo = 55.15 # in kJ/mol
kf = 5.6e7 # in 1/s
kD0 = 4.55e8 # in 1/s
kDinh = 5e9 # in 1/s
kp = 2.654e9 # in 1/s
TPU25 = 7.47 # in umol/m^2/s
DHaTPU = 57.5 # in kJ/mol
DsTPU = 0.79 # in kJ/mol/K
DHdTPU = 246.7 # in kJ/mol
Vrmax = 118.65 # in umol/m^2/s
KmPGA = 5 # in umol/m^2
fpseudo = 0.1
fcyc = 0.1
sigma2 = 0.5
Ib = 100 # in umol/m^2/s
alphabp = 0.66
Ig = 0 # in umol/m^2/s
alphagp = 0.60
Ir = 0 # in umol/m^2/s
alpharp = 0.80
Jmax25 = 139.28 # in umol/m^2/s
DHaJmax = 36.21 # in kJ/mol
DsJmax = 0.69 # in kJ/mol/K
DHdJmax = 215.9 # in kJ/mol
theta = 0.7
Flux0 = 1e-16 # in umol/m^2/s
gamma1 = 0.2
gamma2 = 0.6
gamma3 = 0.2
PhiqEmax = 0.20
ac = 0.27
bc = 0.014 # in mol/umol
fRBmin = 0.48
RCA = 117.37 # in mg/m^2
DHdRCA = 290.2 # in kJ/mol
ToRCA = 300.4 # in K
DHaRCA = 30.0 # in kJ/mol
KaRCA = 10.2 # in mg/m^2
Krca = 8.63e-5 # in 1/s/mg*m^2
KdRB = 6.8e-4 # in 1/s
KiqEp = 1.87e-2 # in 1/s
KdqEp = 2.39e-2 # in 1/s
KiqEz = 1.87e-3 # in 1/s
KdqEz = 2.39e-3 # in 1/s
Iac = 1.6 # in umol/m^2/s
alpharac = 0.05
alphar_alpha25 = 6.55e-3 # in 1/umol*m^2*s
DHaAlphar = 67.32 # in kJ/mol
alpharav = 0.25
thetaalphar = 0.36
Kialpha25 = 1.49e-3 # in 1/s
DHaKalpha = 90.5 # in kJ/mol
DsKalpha =  1.08 # in kJ/mol/K
DHdKalpha = 328 # in kJ/mol
Kdalpha25 = 1.86e-3 # in 1/s
alphab = 0.92
alphag = 0.72
alphared = 0.83
Kinh0 = 1e-7 # in m^2/umol
fprot = 1e-7 # in m^2/umol
zeroKinh = 0 # in m^2/mol
Krep25 = 1.92e-4 # in 1/s
DHaKrep = 160.8 # in kJ/mol
DsKrep = 0.78 # in kJ/mol/K
DHdKrep = 233.23 # in kJ/mol
fR0 = 0.04
alphafR = 0.0025 # in # 1/umol*m^2*s
thetafR = 0.96
KiR = 6.28e-3 # in 1/s
KdR = 7.50e-3 # in 1/s
gw25 = 0.75 # in mol/m^2/s
DHaGw = 70.2 # in kJ/mol
DsGw = 0.32 # in kJ/mol/K
DHdGw = 94.0 # in kJ/mol
kPR = 0.024 # in 1/s
Rm25 = 0.99 # in umol/m^2/s
DHaRm = 56.2 # in kJ/mol
Scm = 7.1
falphaSc = 0.93
Sm = 9.8
gcm25 = 0.39 # in mol/m^2/s
DHaGc = 70.2 # in kJ/mol
DsGc = 0.32 # in kJ/mol/K
DHdGc = 94.0 # in kJ/mol
Ta = 298 # in K
air_pressure = 101 # in kPa
Vref = 1.55e-4 # in m
gbw = 9.2 # in mol/m^2/s
fI0 = 0.39
alphafI = 7.67e-4 # in 1/umol*m^2*s
thetafI = 0.88
es0 = 0.61078 # in kPa
es_k = 17.269
es_Tref = 237.3 # in K
H2OR = 20 # in mmol/mol
D0 = 0.74 # in kPa
gswm = 0.48 # in mol/m^2/s
Kgsi = 1.14e-3 # in 1/s
Kgsd = 1.14e-3 # in 1/s
Flow = 500 # in umol/s
CO2R = 400 # in umol/mol
leaf_surface = 2 # in cm^2
volume_chamber = 80 # in cm^3

# Derived
Kmapp_RuBP = KmRuBP*(1 + PGA/(Vch*KiPGA)) # in mM
fRuBP = 1/(2*RB/Vch)*(RB/Vch + Kmapp_RuBP + RuBP/Vch - np.sqrt((RB/Vch + Kmapp_RuBP + RuBP/Vch)**2 - 4*RB*RuBP/Vch**2))
Kc = Kc25*np.exp((Tl - Tref)*DHaKc/(Tref*R*Tl)) # in 1/s
Kmc = Kmc25*np.exp((Tl - Tref)*DHaKmc/(Tref*R*Tl)) # in umol/mol
Kmo = Kmo25*np.exp((Tl - Tref)*DHaKmo/(Tref*R*Tl)) # in mmol/mol
Ko = Ko25*np.exp((Tl - Tref)*DHaKo/(Tref*R*Tl)) # in 1/s
phi = Kmc*Ko*O2/(Kmo*Kc*Cc) # dimensionless
Fm_a = kf/(kf + kD0)
Fm_d = kf/(kf + kDinh)
Fm = (1 - PSIId)*Fm_a + PSIId*Fm_d
Fo_a =  kf/(kf + kD0 + kp)
Fo_d = kf/(kf + kDinh)
Fo = (1 - PSIId)*Fo_a + PSIId*Fo_d
PhiIIop = (Fm - Fo)/Fm
TPU = TPU25*np.exp((Tl - Tref)*DHaTPU/(Tref*R*Tl))*(1 + np.exp((Tref*DsTPU - DHdTPU)/(Tref*R)))/(1 + np.exp((Tl*DsTPU - DHdTPU)/(Tl*R))) # in umol/m^2/s
VrTPU = 3.0*TPU*(2.0 + 1.5*phi)/(1 - 0.5*phi) # in umol/m^2/s
VrE = fR*Vrmax*PGA/(PGA + KmPGA) # in umol/m^2/s
J2pm =  min(VrTPU,VrE)/(1 - fpseudo/(1 - fcyc))*2.0/(2.0 + 1.5*phi)*(2.0 + 2.0*phi) # in umol/m^2/s
PARaP = Ib*alphabp + Ig*alphagp + Ir*alpharp # in umol/m^2/s
PARaP2 = sigma2*alphar*PARaP # in umol/m^2/s
Jmax = Jmax25*np.exp((Tl - Tref)*DHaJmax/(Tref*R*Tl))*(1 + np.exp((Tref*DsJmax - DHdJmax)/(Tref*R)))/(1 + np.exp((Tl*DsJmax - DHdJmax)/(Tl*R))) # in umol/m^2/s
J2pp = (PhiIIop*PARaP2 + Jmax - np.sqrt((PhiIIop*PARaP2 + Jmax)**2 - 4*PhiIIop*theta*Jmax*PARaP2))/(2*theta) # in umol/m^2/s
qPp = J2pp/PARaP2/PhiIIop if PARaP2 > Flux0 else 1
qPm = J2pm/J2pp*qPp
qPno_qD = min(qPm, qPp)
fqEss = (1 - qPno_qD)
PhiqEss = (fqEss*gamma1 + fqEss*fqEss*gamma2 + fqEss*gamma3)*PhiqEmax
PhiIIoss = PhiIIop - PhiqEss
PhiqE = (fP*gamma1 + fP*fZ*gamma2 + fZ*gamma3)*PhiqEmax
PhiIIo = PhiIIop - PhiqE
J2qE = (1 - (PhiIIoss - PhiIIo)/PhiIIoss)*J2pp if PhiIIo < PhiIIoss else J2pp #  in umol/m^2/s
VrJ = J2qE*(1 - fpseudo/(1 - fcyc))/2.0*(2.0 + 1.5*phi)/(2.0 + 2.0*phi) # in umol/m^2/s
fRBss_nr = min(1.0, ac + bc*Cc)
PAR = Ib + Ig + Ir # in umol/m^2/s
fRCA = DHdRCA*np.exp((Tl - ToRCA)*DHaRCA/(ToRCA*R*Tl))/(DHdRCA - DHaRCA*(1 - np.exp(DHdRCA*(Tl - ToRCA)/(ToRCA*R*Tl))))
fRBmax = RCA*fRCA/(RCA*fRCA + KaRCA)
alphar_alpha = alphar_alpha25*np.exp(-(Tl - Tref)*DHaAlphar/(Tref*R*Tl)) # in 1/umol*m^2*s
alpharss = min(1.0 + Ib/Iac*alpharac, 1.0 + alpharac - (alphar_alpha*(Ib - Iac)  + alpharav -np.sqrt((alphar_alpha*(Ib - Iac) + alpharav)**2 - 4*alphar_alpha*thetaalphar*alpharav*(Ib - Iac)))/(2*thetaalphar))
Kialpha = Kialpha25*np.exp((Tl - Tref)*DHaKalpha/(Tref*R*Tl))*(1 + np.exp((Tref*DsKalpha - DHdKalpha)/(Tref*R)))/(1 + np.exp((Tl*DsKalpha - DHdKalpha)/(Tl*R))) # in 1/s
Kdalpha = Kdalpha25*np.exp((Tl - Tref)*DHaKalpha/(Tref*R*Tl))*(1 + np.exp((Tref*DsKalpha - DHdKalpha)/(Tref*R)))/(1 + np.exp((Tl*DsKalpha - DHdKalpha)/(Tl*R))) # in 1/s
PARa =  Ib*alphab  + Ig*alphag  + Ir*alphared # in umol/m^2/s
Kinh = max(Kinh0 - fprot*PhiqE, zeroKinh) # in m^2/mol
Krep = Krep25*np.exp((Tl - Tref)*DHaKrep/(Tref*R*Tl))*(1 + np.exp((Tref*DsKrep - DHdKrep)/(Tref*R)))/(1 + np.exp((Tl*DsKrep - DHdKrep)/(Tl*R))) # in 1/s
fRss = fR0 + (alphafR*PARa  + (1 - fR0) - np.sqrt((alphafR*PARa + (1 - fR0))**2 - 4*alphafR*thetafR*PARa*(1 - fR0)))/(2*thetafR)
gw = gw25*np.exp((Tl - Tref)*DHaGw/(Tref*R*Tl))*(1 + np.exp((Tref*DsGw - DHdGw)/(Tref*R)))/(1 + np.exp((Tl*DsGw - DHdGw)/(Tl*R))) # in mol/m^2/s
Sc = Scm*falphaSc*alphar
gc = Sc/Sm*gcm25*np.exp((Tl - Tref)*DHaGc/(Tref*R*Tl))*(1 + np.exp((Tref*DsGc - DHdGc)/(Tref*R)))/(1 + np.exp((Tl*DsGc - DHdGc)/(Tl*R))) # in mol/m^2/s
Mv = R*Ta/air_pressure # in dm^3/mol
gsc = gsw/1.56 # in mol/m^2/s
gbc = gbw/1.37 # in mol/m^2/s
fI_b = -(1 + fI0 + alphafI*PAR)
fI_a = thetafI
fI_c = fI0 + alphafI*PAR
fI = (-fI_b - np.sqrt(fI_b**2 - 4*fI_a*fI_c))/(2*fI_a)
es_leaf = es0*np.exp(es_k*(Tl - 273.15)/(es_Tref + (Tl - 273.15))) # in kPa
ea = H2OR*air_pressure # in kPa
VPDleaf = max(es_leaf - ea, 1e-1) # in kPa
fvpd = 1/(1 + VPDleaf/D0)
gss = fI*fvpd*gswm # in mol/m^2/s
transpiration = VPDleaf/(air_pressure - (es_leaf + ea)/2.0)*1/(1/gsw + 1/gbw) # in mmol/m^2/s # Farquhar & Sharkey (1982)

# Rates
Vc = fRB*fRuBP*Kc*RB*Cc/(Cc + Kmc*(1.0 + O2/Kmo)) # in umol/m^2/s
Vr = min(VrJ, VrTPU, VrE) # in umol/m^2/s
v_fP = (fqEss - fP)*KiqEp if fqEss > fP else (fqEss - fP)*KdqEp # 1/s
v_fZ = (fqEss - fZ)*KiqEz if fqEss > fZ else (fqEss - fZ)*KdqEz # 1/s
v_alphar = (alpharss - alphar)*Kialpha if alpharss > alphar else (alpharss - alphar)*Kdalpha # 1/s
v_PSIId = (1 - PSIId)*PARa*alphar*Kinh - PSIId*Krep # in 1/s
v_fR = (fRss - fR)*KiR if fRss > fR else (fRss - fR)*KdR # in 1/s
v_gw = (Ci - Ccyt)*gw
v_Rp = PR*kPR # in umol/m^2/s
v_Rm = Rm25*np.exp((Tl - Tref)*DHaRm/(Tref*R*Tl)) # in umol/m^2/s
v_gv = (Ccyt - Cc)*gc
A = (Ca - Ci)/(1/gsc + 1/gbc) # in umol/m^2/s
v_gsw = (gss - gsw)*Kgsi if gss > gsw else (gss - gsw)*Kgsd # in mol/m^2/s^2
v_ca = -Flow*Ca + Flow*CO2R - leaf_surface*A
v_H2OS = -(Flow + leaf_surface*transpiration)*H2OS + Flow*H2OR + leaf_surface*transpiration

# Derived with Rates
fRBss_r = min(min(VrTPU,VrJ)/(2.0 + 1.5*phi)/(Vc/(fRB*fRuBP)) if Flux0 < PAR else fRBmin, fRBmax)
fRBss = min(fRBss_nr, fRBss_r)

# Rates continued
v_fRB = (fRBss - fRB)*Krca*RCA*fRCA if fRBss > fRB else (fRBss - fRB)*KdRB # 1/s

# ODES

d_PGA_dt = Vc * (2.0 + 1.5*phi) - Vr # in umol/m^2/s
d_RuBP_dt = (1.0 + phi)/(2.0 + 1.5*phi)*Vr - Vc*(1.0 + phi) # in umol/m^2/s
d_fRB_dt = v_fRB
d_fP_dt = v_fP
d_fZ_dt = v_fZ
d_alphar_dt = v_alphar
d_PSIId_dt = v_PSIId
d_fR_dt = v_fR
d_Ccyt_dt = v_gw*Mv/Vref + v_Rp*0.5*Mv/Vref + v_Rm*Mv/Vref - v_gv*Mv/Vref # in umol/mol/s
d_Cc_dt = v_gv*Mv/Vref - Vc*Mv/Vref # in umol/mol/s
d_PR_dt = Vc*phi - v_Rp # in umol/m^2/s
d_Ci_dt= A*Mv/Vref - v_gw*Mv/Vref # in umol/mol/s
d_gsw_dt = v_gsw
d_Ca_dt = v_ca*R*Ta/volume_chamber/air_pressure # in umol/mol/s
d_H2OS_dt = v_H2OS*R*Ta/volume_chamber/air_pressure # in mmol/mol/s

print(f"{Mv/Vref=}, {Mv=}, {Vref=}")

from decimal import Decimal
import pandas as pd

def get_rhs():
    res = pd.Series({
        "3PGA": d_PGA_dt,
        "RUBP": d_RuBP_dt,
        "fRB": d_fRB_dt,
        "fP": d_fP_dt,
        "fZ": d_fZ_dt,
        "alphar": d_alphar_dt,
        "PSII_damaged": d_PSIId_dt,
        "fR": d_fR_dt,
        "CO2 (dissolved)_cytosol": d_Ccyt_dt,
        "CO2 (dissolved)_chloroplast": d_Cc_dt,
        "PR": d_PR_dt,
        "Ci": d_Ci_dt,
        "gsw": d_gsw_dt,
        "Ca": d_Ca_dt,
        "H2OS": d_H2OS_dt,
    })
    
    return res

def get_fluxes():
    res = pd.Series({
        "Vc": Vc,
        "Vr": Vr,
        "v_fP": v_fP,
        "v_fZ": v_fZ,
        "v_alphar": v_alphar,
        "v_PSIId": v_PSIId,
        "v_fR": v_fR,
        "v_gw": v_gw,
        "v_Rp": v_Rp,
        "v_Rm": v_Rm,
        "v_gv": v_gv,
        "A": A,
        "v_gsw": v_gsw,
        "v_ca": v_ca,
        "v_H2OS": v_H2OS,
        "v_fRB": v_fRB,
    })
    
    return res

from model import get_morales2018

for s in ["rhs", "fluxes"]:
    if s == "rhs":
        mxlpy_res = get_morales2018().get_right_hand_side()
        this_res = get_rhs()
    elif s == "fluxes":
        mxlpy_res = get_morales2018().get_fluxes()
        this_res = get_fluxes()
    else:
        raise ValueError(f"Unknown section '{s}'")
        
    res = mxlpy_res - this_res
    res.name = s
    
    print(res[res != 0])
    print()
    
print(get_morales2018().get_stoichiometries_of_variable("CO2 (dissolved)_cytosol"))

print(f"{alpharss=}, {alphar=}, {Kialpha=}, {Kdalpha=}")
print(f"{Ib=}, {Iac=}, {alpharac=}, {alphar_alpha=}, {alpharav=}, {thetaalphar=}")

print(f"{alphar_alpha25=}, {Tl=}, {Tref=}, {DHaAlphar=}, {R=}")

from mxlpy import Simulator

s = Simulator(get_morales2018())

s.simulate(1)

