from mxlpy import Model, Variable, Parameter, units, Derived
import mxlbricks.names as n
from mxlbricks.fns import minus, value, neg_div, div
from added_names import add_temp, energy_activation, k, entropy, energy_deactivation
from added_fns import half_div
import numpy as np

# Derived functions

def _Kmapp_RuBP(KmRuBP: float, PGA: float, Vch: float, KiPGA: float) -> float:
    return KmRuBP*(1 + PGA/(Vch*KiPGA))

def _fRuBP(RB: float, Vch: float, Kmapp_RuBP: float, RuBP: float) -> float:
    return 1/(2*RB/Vch)*(RB/Vch + Kmapp_RuBP + RuBP/Vch - np.sqrt((RB/Vch + Kmapp_RuBP + RuBP/Vch)**2 - 4*RB*RuBP/Vch**2))

def _phi(Kmc: float, Ko: float, O2: float, Kmo: float, Kc: float, Cc: float) -> float:
    return Kmc*Ko*O2/(Kmo*Kc*Cc)

def simple_arrhenius(param_25: float, Ea: float, Tref: float, R: float, Tl: float) -> float:
    # print(f"{param_25=}, {Ea=}, {Tref=}, {R=}, {Tl=}")
    return param_25*np.exp((Tl - Tref)*Ea/(Tref*R*Tl))

def _Fm_a(kf: float, kD0: float) -> float:
    return kf/(kf + kD0)

def _Fm_d(kf: float, kDinh: float) -> float:
    return kf/(kf + kDinh)

def _Fm(Fm_a: float, Fm_d: float, PSIId: float) -> float:
    return (1 - PSIId)*Fm_a + PSIId*Fm_d

def _Fo_a(kf: float, kD0: float, kp: float) -> float:
    return kf/(kf + kD0 + kp)

def _Fo_d(kf: float, kDinh: float) -> float:
    return kf/(kf + kDinh)

def _Fo(Fo_a: float, Fo_d: float, PSIId: float) -> float:
    return (1 - PSIId)*Fo_a + PSIId*Fo_d

def _PhiIIop(Fm: float, Fo: float) -> float:
    return (Fm - Fo)/Fm

def extended_arrhenius(param_25: float, Ea: float, Entropy: float, Eda: float, Tref: float, R: float, Tl: float) -> float:
    return param_25*np.exp((Tl - Tref)*Ea/(Tref*R*Tl))*(1 + np.exp((Tref*Entropy - Eda)/(Tref*R)))/(1 + np.exp((Tl*Entropy - Eda)/(Tl*R)))

def _VrTPU(TPU: float, phi: float) -> float:
    return 3.0*TPU*(2.0 + 1.5*phi)/(1 - 0.5*phi)

def _VrE(fR: float, Vrmax: float, PGA: float, KmPGA: float) -> float:
    return fR*Vrmax*PGA/(PGA + KmPGA)

def _J2pm(VrTPU: float, VrE: float, fpseudo: float, fcyc: float, phi: float) -> float:
    return min(VrTPU,VrE)/(1 - fpseudo/(1 - fcyc))*2.0/(2.0 + 1.5*phi)*(2.0 + 2.0*phi)

def _PARa(Ib: float, alphabp: float, Ig: float, alphagp: float, Ir: float, alpharp: float) -> float:
    return Ib*alphabp + Ig*alphagp + Ir*alpharp

def _PARaP2(sigma2: float, alphar: float, PARaP: float) -> float:
    return sigma2*alphar*PARaP

def _J2pp(PhiIIop: float, PARaP2: float, Jmax: float, theta: float) -> float:
    return (PhiIIop*PARaP2 + Jmax - np.sqrt((PhiIIop*PARaP2 + Jmax)**2 - 4*PhiIIop*theta*Jmax*PARaP2))/(2*theta)

def _qPp(PARaP2: float, PhiIIop: float, J2pp: float, Flux0: float) -> float:
    return J2pp/PARaP2/PhiIIop if PARaP2 > Flux0 else 1

def _qPm(J2pm: float, J2pp: float, qPp: float) -> float:
    return J2pm/J2pp*qPp

def _qPno_qD(qPm: float, qPp: float) -> float:
    return min(qPm, qPp)

def _fqEss(qPno_qD: float) -> float:
    return (1 - qPno_qD)

def _PhiqEss(fqEss: float, gamma1: float, gamma2: float, gamma3: float, PhiqEmax: float) -> float:
    return (fqEss*gamma1 + fqEss*fqEss*gamma2 + fqEss*gamma3)*PhiqEmax # TODO: chekc if correct

def _PhiqE(fP: float, fZ: float, gamma1: float, gamma2: float, gamma3: float, PhiqEmax: float) -> float:
    return (fP*gamma1 + fP*fZ*gamma2 + fZ*gamma3)*PhiqEmax

def _J2qE(PhiIIoss: float, PhiIIo: float, J2pp: float) -> float:
    return (1 - (PhiIIoss - PhiIIo)/PhiIIoss)*J2pp if PhiIIo < PhiIIoss else J2pp

def _VrJ(J2qE: float, fpseudo: float, fcyc: float, phi: float) -> float:
    return J2qE*(1 - fpseudo/(1 - fcyc))/2.0*(2.0 + 1.5*phi)/(2.0 + 2.0*phi)

def _fRBss_nr(ac: float, bc: float, Cc: float) -> float:
    return min(1.0, ac + bc*Cc)

def _PAR(Ib: float, Ig: float, Ir: float) -> float:
    return Ib + Ig + Ir

def arrhenius_optimal(Eda: float, Ea: float, Tl: float, To: float, R: float) -> float:
    return Eda*np.exp((Tl - To)*Ea/(To*R*Tl))/(Eda - Ea*(1 - np.exp(Eda*(Tl - To)/(To*R*Tl))))

def _fRBmax(RCA: float, fRCA: float, KaRCA: float) -> float:
    return RCA*fRCA/(RCA*fRCA + KaRCA)

def _alpharss(Ib: float, Iac:float, alpharac: float, alphar_alpha: float, alpharav: float, thetaalphar: float) -> float:
    # print(f"{Ib=}, {Iac=}, {alpharac=}, {alphar_alpha=}, {alpharav=}, {thetaalphar=}")
    return min(1.0 + Ib/Iac*alpharac, 1.0 + alpharac - (alphar_alpha*(Ib - Iac)  + alpharav -np.sqrt((alphar_alpha*(Ib - Iac) + alpharav)**2 - 4*alphar_alpha*thetaalphar*alpharav*(Ib - Iac)))/(2*thetaalphar))

def _Kinh(Kinh0: float, fprot: float, PhiqE: float, zeroKinh: float) -> float:
    return max(Kinh0 - fprot*PhiqE, zeroKinh)

def _fRss_nr(fR0: float, alphafR: float, PARa: float, thetafR: float) -> float:
    return fR0 + (alphafR*PARa  + (1 - fR0) - np.sqrt((alphafR*PARa + (1 - fR0))**2 - 4*alphafR*thetafR*PARa*(1 - fR0)))/(2*thetafR)

def _Sc(Scm: float, falphaSc: float, alphar: float) -> float:
    return Scm*falphaSc*alphar

def _Mv(R: float, Ta: float, air_pressure: float) -> float:
    return R*Ta/air_pressure

def _gsc(gsw: float) -> float:
    return gsw/1.56

def _gbc(gbw: float) -> float:
    return gbw/1.37

def _fI_b(fI0: float, alphafI: float, PAR: float) -> float:
    return -(1 + fI0 + alphafI*PAR)

def _fI_c(fI0: float, alphafI: float, PAR: float) -> float:
    return fI0 + alphafI*PAR

def _fI(fI_b: float, fI_a: float, fI_c: float) -> float:
    return (-fI_b - np.sqrt(fI_b**2 - 4*fI_a*fI_c))/(2*fI_a)

def _es_leaf(es0: float, es_k: float, Tl: float, es_Tref: float) -> float:
    return es0*np.exp(es_k*(Tl - 273.15)/(es_Tref + (Tl - 273.15)))

def _ea(H2OR: float, air_pressure: float) -> float:
    return H2OR*air_pressure

def _VPDleaf(es_leaf: float, ea: float) -> float:
    return max(es_leaf - ea, 1e-1)

def _fvpd(VPDleaf: float, D0: float) -> float:
    return 1/(1 + VPDleaf/D0)

def _gss(fI: float, fvpd: float, gswm: float) -> float:
    return fI*fvpd*gswm

def _transpiration(VPDleaf: float, air_pressure: float, es_leaf: float, ea: float, gsw: float, gbw: float) -> float:
    return VPDleaf/(air_pressure - (es_leaf + ea)/2.0)*1/(1/gsw + 1/gbw)

# Rate Functions

def _Vc(fRB: float, fRuBP: float, Kc: float, RB: float, Cc: float, Kmc: float, O2: float, Kmo: float) -> float:
    return fRB*fRuBP*Kc*RB*Cc/(Cc + Kmc*(1.0 + O2/Kmo))

def _Vr(VrJ: float, VrTPU: float, VrE: float) -> float:
    return min(VrJ, VrTPU, VrE)

def inc_dec_rate(val_ss: float, val: float, Ki: float, Kd: float) -> float:
    # print(f"{val_ss=}, {val=}, {Ki=}, {Kd=}")
    return (val_ss - val) * Ki if val_ss > val else (val_ss - val) * Kd

def _v_PSIId(PSIId: float, PARa: float, alphar: float, Kinh: float, Krep: float) -> float:
    return (1 - PSIId)*PARa*alphar*Kinh - PSIId*Krep

def _v_g(C1: float, C2: float, g: float) -> float:
    return (C1 - C2)*g

def _v_Rp(PR: float, kPR: float) -> float:
    return 0.5*PR*kPR

def _A(Ca: float, Ci: float, gsc: float, gbc: float) -> float:
    return (Ca - Ci)/(1/gsc + 1/gbc)

def _v_ca(Flow: float, Ca: float, CO2R: float, leaf_surface: float, A: float) -> float:
    return -Flow*Ca + Flow*CO2R - leaf_surface*A

def _v_H2OS(Flow: float, H2OS: float, H2OR: float, leaf_surface: float, transpiration: float) -> float:
    return -(Flow + leaf_surface*transpiration)*H2OS + Flow*H2OR + leaf_surface*transpiration

def _v_fRB(VrTPU: float, VrJ: float, phi: float, Vc: float, fRB: float, fRuBP: float, Flux0: float, PAR: float, fRBmin: float, fRBmax: float, fRBss_nr: float, Krca: float, RCA: float, fRCA: float, KdRB: float) -> float:
    fRBss_r = min(min(VrTPU,VrJ)/(2.0 + 1.5*phi)/(Vc/(fRB*fRuBP)) if Flux0 < PAR else fRBmin, fRBmax)
    fRBss = min(fRBss_nr, fRBss_r)
    return (fRBss - fRB)*Krca*RCA*fRCA if fRBss > fRB else (fRBss - fRB)*KdRB

# Derived Stoics

def _stoic_vc_pga(phi: float) -> float:
    return 2.0 + 1.5*phi

def _stoic_vc_rubp(phi: float) -> float:
    return -(1.0 + phi)

def _stoic_vr_rubp(phi: float) -> float:
    return (1.0 + phi)/(2.0 + 1.5*phi)

def _stoic_chamber(R: float, Ta: float, volume_chamber: float, air_pressure: float) -> float:
    return R*Ta/volume_chamber/air_pressure

# fRB -> Ract
# PSIId -> n.ps2("_damaged")
# Ccyt -> n.co2("_cytosol")
# Cc -> n.co2("_chloroplast")
# Vc -> n.rubisco_carboxylase()

def get_morales2018() -> Model:
    """
    Morales A, Kaiser E, Yin X, et al. Dynamic modelling of limitations on improving leaf CO2 assimilation under fluctuating irradiance. Plant Cell Environ. 2018; 41: 589–604. https://doi.org/10.1111/pce.13119
    """
    
    model = Model()
    
    unit_mumol_per_sqm = units.mumol / units.sqm
    unit_mumol_per_mol = units.mumol / units.mol
    unit_mM = units.milli * units.mol / units.liter
    unit_kJ = 1000 * units.joule 
    
    model.add_variables({
        n.pga(): Variable(50, unit=unit_mumol_per_sqm),
        n.rubp(): Variable(50, unit=unit_mumol_per_sqm),
        "fRB": Variable(0.25, unit=units.dimensionless), # fRB in paper
        "fP": Variable(0, unit=units.dimensionless),   # fP in paper
        "fZ": Variable(0, unit=units.dimensionless),   # fZ in paper
        "alphar": Variable(1, unit=units.dimensionless), # alphar in paper
        n.ps2("_damaged"): Variable(0, unit=units.dimensionless), # PSIId in paper
        "fR": Variable(0, unit=units.dimensionless),   # fR in paper
        n.co2("_cytosol"): Variable(380, unit=unit_mumol_per_mol), # Ccyt in paper
        n.co2("_chloroplast"): Variable(380, unit=unit_mumol_per_mol), # Cc in paper
        "PR": Variable(0, unit=unit_mumol_per_mol),  # PR in paper
        "Ci": Variable(380, unit=unit_mumol_per_mol), # Ci in paper
        "gsw": Variable(0.09, unit=unit_mumol_per_sqm / units.second), # gsw in paper
        "Ca": Variable(380, unit=unit_mumol_per_mol),  # Ca in paper
        "H2OS": Variable(20, unit=units.mmol / units.mol), # H2OS in paper
    })
    
    model.add_parameters({
        "RB": Parameter(15.90, unit=unit_mumol_per_sqm, source="6"), # RB in paper #TODO: different in code and info?
        "Vch": Parameter(10, unit=units.milli * units.liter / units.sqm, source="29"), # Vst in paper Vch in Code
        "KmRuBP": Parameter(0.02, unit=unit_mM, source="26"), # KmRuBP in paper
        "KiPGA": Parameter(0.84, unit=unit_mM, source="24"), # KiPGA in paper
        "Kc25": Parameter(4.16, unit=units.per_second, source="10"), # Kc25 in paper #TODO: 4.15 in paper table
        "T": Parameter(298, unit=units.kelvin, source=""), # Tl in paper
        "Tref": Parameter(298.15, unit=units.kelvin, source=""), # Tref in paper
        "DHaKc": Parameter(41.82, unit=1000 / units.joule / units.mol, source="10"), # DHaKc in paper #TODO: kJ in code and J in Paper
        "R": Parameter(8.31, unit=units.joule / (units.mol * units.kelvin), source=""), # R in paper
        "Kmc25": Parameter(261.7, unit=unit_mumol_per_mol, source="10"), # Kmc25 in paper #TODO: CHekc units
        "DHaKmc": Parameter(49.43, unit=1000 / units.joule / units.mol, source="10"), # DHaKmc in paper #TODO: kJ in code and J in Paper
        "O2": Parameter(210, unit=units.mmol / units.mol, source=None), # O2 in paper
        "Kmo25": Parameter(198.5, unit=units.mmol / units.mol, source="10"), # Kmo25 in paper #TODO: Check units
        "DHaKmo": Parameter(29.08, unit=1000 / units.joule / units.mol, source="10"), # DHaKmo in paper #TODO: kJ in code and J in
        "Ko25": Parameter(1.26, unit=units.per_second, source="10"), # Ko25 in paper #TODO: 0.83 in paper table
        "DHaKo": Parameter(55.15, unit=1000 / units.joule / units.mol, source="10"), # DHaKo in paper #TODO: kJ in code and J in Paper
        "kf": Parameter(5.6e7, unit=units.per_second, source="22"), # kf in paper TODO: 6.9e7 in paper table
        "kD0": Parameter(4.55e8, unit=units.per_second, source="22"), # kD0 in paper
        "kDinh": Parameter(5e9, unit=units.per_second, source="23"), # kDinh in paper
        "kp": Parameter(2.654e9, unit=units.per_second, source="22"), # kp in paper TODO: 2.6e9 in paper table
        "TPU25": Parameter(7.47, unit=unit_mumol_per_sqm / units.second, source="6"), # TPU25 in paper TODO: 8.14e-6 in paper table
        "DHaTPU": Parameter(57.5, unit=1000 / units.joule / units.mol, source="15"), # DHaTPU in paper TODO: 5.28e4 j/mol in paper table kJ in code and J in Paper
        "DsTPU": Parameter(0.79, unit=1000 / units.joule / (units.mol * units.kelvin), source="15"), # DsTPU in paper TODO: 1199 J/mol/K in paper table kJ in code
        "DHdTPU": Parameter(246.7, unit=1000 / units.joule / units.mol, source="15"), # DHdTPU in paper TODO: 3.71e5 j/mol in paper table kJ in code and J in Paper
        "Vrmax": Parameter(118.65, unit=unit_mumol_per_sqm / units.second, source="31"), # Vrmax in paper TODO: 4.39e-4 in paper table
        "KmPGA": Parameter(5, unit=unit_mumol_per_sqm, source="26"), # KmPGA in paper
        "fpseudo": Parameter(0.1, unit=units.dimensionless, source=""), # fpseudo in paper
        "fcyc": Parameter(0.1, unit=units.dimensionless, source=""), # fcyc in paper
        "sigma2": Parameter(0.5, unit=units.dimensionless, source=""),
        "Ib": Parameter(100, unit=unit_mumol_per_sqm / units.second, source=""),
        "alphabp": Parameter(0.66, unit=units.dimensionless, source=""),
        "Ig": Parameter(0, unit=unit_mumol_per_sqm / units.second, source=""),
        "alphagp": Parameter(0.60, unit=units.dimensionless, source=""),
        "Ir": Parameter(0, unit=unit_mumol_per_sqm / units.second, source=""),
        "alpharp": Parameter(0.80, unit=units.dimensionless, source=""),
        "Jmax25": Parameter(139.28, unit=unit_mumol_per_sqm / units.second, source=""),
        "DHaJmax": Parameter(36.21, unit=1000 / units.joule / units.mol, source=""),
        "DsJmax": Parameter(0.69, unit=1000 / (units.joule * units.mol * units.kelvin), source=""),
        "DHdJmax": Parameter(215.9, unit=1000 / units.joule / units.mol, source=""),
        "theta": Parameter(0.7, unit=units.dimensionless, source=""),
        "Flux0": Parameter(1e-16, unit=unit_mumol_per_sqm / units.second, source=""),
        "gamma1": Parameter(0.2, unit=units.dimensionless, source=""),
        "gamma2": Parameter(0.6, unit=units.dimensionless, source=""),
        "gamma3": Parameter(0.2, unit=units.dimensionless, source=""),
        "PhiqEmax": Parameter(0.20, unit=units.dimensionless, source=""),
        "ac": Parameter(0.27, unit=units.dimensionless, source=""),
        "bc": Parameter(0.014, unit=units.mol / units.mumol, source=""),
        "fRBmin": Parameter(0.48, unit=units.dimensionless, source=""),
        "RCA": Parameter(117.37, unit=1000 *units.gram / units.sqm, source=""),
        "DHdRCA": Parameter(290.2, unit=1000 / units.joule / units.mol, source=""),
        "ToRCA": Parameter(300.4, unit=units.kelvin, source=""),
        "DHaRCA": Parameter(30.0, unit=1000 / units.joule / units.mol, source=""),
        "KaRCA": Parameter(10.2, unit=1000 * units.gram / units.sqm, source=""),
        "Krca": Parameter(8.63e-5, unit=units.per_second / (1000 * units.gram / units.sqm), source=""),
        "KdRB": Parameter(6.8e-4, unit=units.per_second / (1000 * units.gram / units.sqm), source=""),
        "KiqEp": Parameter(1.87e-2, unit=units.per_second, source=""),
        "KdqEp": Parameter(2.39e-2, unit=units.per_second, source=""),
        "KiqEz": Parameter(1.87e-3, unit=units.per_second, source=""),
        "KdqEz": Parameter(2.39e-3, unit=units.per_second, source=""),
        "Iac": Parameter(1.6, unit=unit_mumol_per_sqm / units.second, source=""),
        "alpharac": Parameter(0.05, unit=units.dimensionless, source=""),
        "alphar_alpha25": Parameter(6.55e-3, unit=units.per_second / (units.mumol * units.sqm * units.second), source=""),
        "DHaAlphar": Parameter(67.32, unit=1000 / units.joule / units.mol, source=""),
        "alpharav": Parameter(0.25, unit=units.dimensionless, source=""),
        "thetaalphar": Parameter(0.36, unit=units.dimensionless, source=""),
        "Kialpha25": Parameter(1.49e-3, unit=units.per_second, source=""),
        "DHaKalpha": Parameter(90.5, unit=1000 / units.joule / units.mol, source=""),
        "DsKalpha": Parameter(1.08, unit=1000 / (units.joule * units.mol * units.kelvin), source=""),
        "DHdKalpha": Parameter(328, unit=1000 / units.joule / units.mol, source=""),
        "Kdalpha25": Parameter(1.86e-3, unit=units.per_second, source=""),
        "alphab": Parameter(0.92, unit=units.dimensionless, source=""),
        "alphag": Parameter(0.72, unit=units.dimensionless, source=""),
        "alphared": Parameter(0.83, unit=units.dimensionless, source=""),
        "Kinh0": Parameter(1e-7, unit=units.sqm / units.mumol, source=""),
        "fprot": Parameter(1e-7, unit=units.sqm / units.mumol, source=""),
        "zeroKinh": Parameter(0.0, unit=units.sqm / units.mumol, source=""),
        "Krep25": Parameter(1.92e-4, unit=unit_mumol_per_mol, source=""),
        "DHaKrep": Parameter(160.8, unit=1000 / units.joule / units.mol, source=""),
        "DsKrep": Parameter(0.78, unit=1000 / units.joule / (units.mol * units.kelvin), source=""),
        "DHdKrep": Parameter(233.23, unit=1000 / units.joule / units.mol, source=""),
        "fR0": Parameter(0.04, unit=units.dimensionless, source=""),
        "alphafR": Parameter(0.0025, unit=units.per_second / (units.mumol * units.sqm * units.second), source=""),
        "thetafR": Parameter(0.96, unit=units.dimensionless, source=""),
        "KiR": Parameter(6.28e-3, unit=units.per_second, source=""),
        "KdR": Parameter(7.50e-3, unit=units.per_second, source=""),
        "gw25": Parameter(0.75, unit=units.mol / (units.sqm * units.second), source=""),
        "DHaGw": Parameter(70.2, unit=1000 / units.joule / units.mol, source=""),
        "DsGw": Parameter(0.32, unit=1000 / units.joule / (units.mol * units.kelvin), source=""),
        "DHdGw": Parameter(94.0, unit=1000 / units.joule / units.mol, source=""),
        "kPR": Parameter(0.024, unit=units.per_second, source=""),
        "Rm25": Parameter(0.99, unit=unit_mumol_per_sqm / units.second, source=""),
        "DHaRm": Parameter(56.2, unit=1000 / units.joule / units.mol, source=""),
        "Scm": Parameter(7.1, unit=units.dimensionless, source=""),
        "falphaSc": Parameter(0.93, unit=units.dimensionless, source=""),
        "Sm": Parameter(9.8, unit=units.dimensionless, source=""),
        "gcm25": Parameter(0.39, unit=units.mol / (units.sqm * units.second), source=""),
        "DHaGc": Parameter(70.2, unit=1000 / units.joule / units.mol, source=""),
        "DsGc": Parameter(0.32, unit=1000 / units.joule / (units.mol * units.kelvin), source=""),
        "DHdGc": Parameter(94.0, unit=1000 / units.joule / units.mol, source=""),
        "Ta": Parameter(298, unit=units.kelvin, source=""),
        "air_pressure": Parameter(101, unit=1000 / units.pascal, source=""),
        "Vref": Parameter(1.55e-4),
        "gbw": Parameter(9.2, unit=units.mol / (units.sqm * units.second), source=""),
        "fI0": Parameter(0.39, unit=units.dimensionless, source=""),
        "alphafI": Parameter(7.67e-4, unit=units.per_second / (units.mumol * units.sqm * units.second), source=""),
        "thetafI": Parameter(0.88, unit=units.dimensionless, source=""),
        "es0": Parameter(0.61078, unit=1000 / units.pascal, source=""),
        "es_k": Parameter(17.269, unit=units.dimensionless, source=""),
        "es_Tref": Parameter(237.3, unit=units.kelvin, source=""),
        "H2OR": Parameter(20, unit=units.mmol / units.mol, source=""),
        "D0": Parameter(0.74, unit=1000 / units.pascal, source=""),
        "gswm": Parameter(0.48, unit=units.mol / (units.sqm * units.second), source=""),
        "Kgsi": Parameter(1.14e-3, unit=units.per_second, source=""),
        "Kgsd": Parameter(1.14e-3, unit=units.per_second, source=""),
        "Flow": Parameter(500, unit=units.mumol / units.second, source=""),
        "CO2R": Parameter(400, unit=unit_mumol_per_mol, source=""),
        "leaf_surface": Parameter(2),
        "volume_chamber": Parameter(80),
    })
    
    # Derived
    model.add_derived(
        name="Kmapp_RuBP",
        fn=_Kmapp_RuBP,
        args=["KmRuBP", n.pga(), "Vch", "KiPGA"],
        unit=unit_mM
    )
    
    model.add_derived(
        name="fRuBP",
        fn=_fRuBP,
        args=["RB", "Vch", "Kmapp_RuBP", n.rubp()],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="Kc",
        fn=simple_arrhenius,
        args=["Kc25", "DHaKc", "Tref", "R", "T"],
        unit=units.per_second
    )
    
    model.add_derived(
        name="Kmc",
        fn=simple_arrhenius,
        args=["Kmc25", "DHaKmc", "Tref", "R", "T"],
        unit=unit_mumol_per_mol
    )
    
    model.add_derived(
        name="Kmo",
        fn=simple_arrhenius,
        args=["Kmo25", "DHaKmo", "Tref", "R", "T"],
        unit=units.mmol / units.mol
    )
    
    model.add_derived(
        name="Ko",
        fn=simple_arrhenius,
        args=["Ko25", "DHaKo", "Tref", "R", "T"],
        unit=units.per_second
    )
    
    model.add_derived(
        name="phi",
        fn=_phi,
        args=["Kmc", "Ko", "O2", "Kmo", "Kc", n.co2("_chloroplast")],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="Fm_a",
        fn=_Fm_a,
        args=["kf", "kD0"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="Fm_d",
        fn=_Fm_d,
        args=["kf", "kDinh"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="Fm",
        fn=_Fm,
        args=["Fm_a", "Fm_d", n.ps2("_damaged")],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="Fo_a",
        fn=_Fo_a,
        args=["kf", "kD0", "kp"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="Fo_d",
        fn=_Fo_d,
        args=["kf", "kDinh"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="Fo",
        fn=_Fo,
        args=["Fo_a", "Fo_d", n.ps2("_damaged")],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="PhiIIop",
        fn=_PhiIIop,
        args=["Fm", "Fo"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="TPU",
        fn=extended_arrhenius,
        args=["TPU25", "DHaTPU", "DsTPU", "DHdTPU", "Tref", "R", "T"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="VrTPU",
        fn=_VrTPU,
        args=["TPU", "phi"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="VrE",
        fn=_VrE,
        args=["fR", "Vrmax", n.pga(), "KmPGA"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="J2pm",
        fn=_J2pm,
        args=["VrTPU", "VrE", "fpseudo", "fcyc", "phi"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="PARaP",
        fn=_PARa,
        args=["Ib", "alphabp", "Ig", "alphagp", "Ir", "alpharp"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="PARaP2",
        fn=_PARaP2,
        args=["sigma2", "alphar", "PARaP"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="Jmax",
        fn=extended_arrhenius,
        args=["Jmax25", "DHaJmax", "DsJmax", "DHdJmax", "Tref", "R", "T"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="J2pp",
        fn=_J2pp,
        args=["PhiIIop", "PARaP2", "Jmax", "theta"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="qPp",
        fn=_qPp,
        args=["PARaP2", "PhiIIop", "J2pp", "Flux0"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="qPm",
        fn=_qPm,
        args=["J2pm", "J2pp", "qPp"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="qPno_qD",
        fn=_qPno_qD,
        args=["qPm", "qPp"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="fqEss",
        fn=_fqEss,
        args=["qPno_qD"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="PhiqEss",
        fn=_PhiqEss,
        args=["fqEss", "gamma1", "gamma2", "gamma3", "PhiqEmax"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="PhiIIoss",
        fn=minus,
        args=["PhiIIop", "PhiqEss"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="PhiqE",
        fn=_PhiqE,
        args=["fP", "fZ", "gamma1", "gamma2", "gamma3", "PhiqEmax"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="PhiIIo",
        fn=minus,
        args=["PhiIIop", "PhiqE"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="J2qE",
        fn=_J2qE,
        args=["PhiIIoss", "PhiIIo", "J2pp"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="VrJ",
        fn=_VrJ,
        args=["J2qE", "fpseudo", "fcyc", "phi"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="fRBss_nr",
        fn=_fRBss_nr,
        args=["ac", "bc", n.co2("_chloroplast")],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="PAR",
        fn=_PAR,
        args=["Ib", "Ig", "Ir"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="fRCA",
        fn=arrhenius_optimal,
        args=["DHdRCA", "DHaRCA", "T", "ToRCA", "R"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="fRBmax",
        fn=_fRBmax,
        args=["RCA", "fRCA", "KaRCA"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="alphar_alpha",
        fn=simple_arrhenius,
        args=["alphar_alpha25", "DHaAlphar", "Tref", "R", "T"],
        unit=units.per_second / (units.mumol * units.sqm * units.second)
    )
    
    model.add_derived(
        name="alpharss",
        fn=_alpharss,
        args=["Ib", "Iac", "alpharac", "alphar_alpha", "alpharav", "thetaalphar"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="Kialpha",
        fn=extended_arrhenius,
        args=["Kialpha25", "DHaKalpha", "DsKalpha", "DHdKalpha", "Tref", "R", "T"],
        unit=units.per_second
    )
    
    model.add_derived(
        name="Kdalpha",
        fn=extended_arrhenius,
        args=["Kdalpha25", "DHaKalpha", "DsKalpha", "DHdKalpha", "Tref", "R", "T"],
        unit=units.per_second
    )
    
    model.add_derived(
        name="PARa",
        fn=_PARa,
        args=["Ib", "alphab", "Ig", "alphag", "Ir", "alphared"],
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_derived(
        name="Kinh",
        fn=_Kinh,
        args=["Kinh0", "fprot", "PhiqE", "zeroKinh"],
        unit=units.sqm / units.mumol
    )
    
    model.add_derived(
        name="Krep",
        fn=extended_arrhenius,
        args=["Krep25", "DHaKrep", "DsKrep", "DHdKrep", "Tref", "R", "T"],
        unit=unit_mumol_per_mol
    )
    
    model.add_derived(
        name="fRss",
        fn=_fRss_nr,
        args=["fR0", "alphafR", "PARa", "thetafR"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="gw",
        fn=extended_arrhenius,
        args=["gw25", "DHaGw", "DsGw", "DHdGw", "Tref", "R", "T"],
        unit=units.mol / (units.sqm * units.second)
    )
    
    model.add_derived(
        name="Sc",
        fn=_Sc,
        args=["Scm", "falphaSc", "alphar" ],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="gc",
        fn=extended_arrhenius,
        args=["gcm25", "DHaGc", "DsGc", "DHdGc", "Tref", "R", "T"],
        unit=units.mol / (units.sqm * units.second)
    )
    
    model.add_derived(
        name="Mv",
        fn=_Mv,
        args=["R", "Ta", "air_pressure"],
    )
    
    model.add_derived(
        name="gsc",
        fn=_gsc,
        args=["gsw"],
        unit=units.mol / (units.sqm * units.second)
    )
    
    model.add_derived(
        name="gbc",
        fn=_gbc,
        args=["gbw"],
        unit=units.mol / (units.sqm * units.second)
    )
    
    model.add_derived(
        name="fI_b",
        fn=_fI_b,
        args=["fI0", "alphafI", "PAR"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="fI_a",
        fn=value,
        args=["thetafI"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="fI_c",
        fn=_fI_c,
        args=["fI0", "alphafI", "PAR"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="fI",
        fn=_fI,
        args=["fI_b", "fI_a", "fI_c"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="es_leaf",
        fn=_es_leaf,
        args=["es0", "es_k", "T", "es_Tref"],
        unit=1000 / units.pascal
    )
    
    model.add_derived(
        name="ea",
        fn=_ea,
        args=["H2OR", "air_pressure"],
        unit=1000 / units.pascal
    )
    
    model.add_derived(
        name="VPDleaf",
        fn=_VPDleaf,
        args=["es_leaf", "ea"],
        unit=1000 / units.pascal
    )
    
    model.add_derived(
        name="fvpd",
        fn=_fvpd,
        args=["VPDleaf", "D0"],
        unit=units.dimensionless
    )
    
    model.add_derived(
        name="gss",
        fn=_gss,
        args=["fI", "fvpd", "gswm"],
        unit=units.mol / (units.sqm * units.second)
    )
    
    model.add_derived(
        name="transpiration",
        fn=_transpiration,
        args=["VPDleaf", "air_pressure", "es_leaf", "ea", "gsw", "gbw"],
        unit=units.mmol / (units.sqm * units.second)
    )
    
    # Rates
    
    model.add_reaction(
        name="Vc",
        fn=_Vc,
        args=["fRB", "fRuBP", "Kc", "RB", n.co2("_chloroplast"), "Kmc", "O2", "Kmo"],
        stoichiometry={
            n.pga(): Derived(fn=_stoic_vc_pga, args=["phi"]),
            n.rubp(): Derived(fn=_stoic_vc_rubp, args=["phi"]),
            n.co2("_chloroplast"): Derived(fn=neg_div, args=["Mv", "Vref"]),
            "PR": Derived(fn=value, args=["phi"])
        },
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_reaction(
        name="Vr",
        fn=_Vr,
        args=["VrJ", "VrTPU", "VrE"],
        stoichiometry={
            n.pga(): -1,
            n.rubp(): Derived(fn=_stoic_vr_rubp, args=["phi"])
        },
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_reaction(
        name="v_fP",
        fn=inc_dec_rate,
        args=["fqEss", "fP", "KiqEp", "KdqEp"],
        stoichiometry={
            "fP": 1
        },
        unit=units.per_second
    )
    
    model.add_reaction(
        name="v_fZ",
        fn=inc_dec_rate,
        args=["fqEss", "fZ", "KiqEz", "KdqEz"],
        stoichiometry={
            "fZ": 1
        },
        unit=units.per_second
    )
    
    model.add_reaction(
        name="v_alphar",
        fn=inc_dec_rate,
        args=["alpharss", "alphar", "Kialpha", "Kdalpha"],
        stoichiometry={
            "alphar": 1
        },
        unit=units.per_second
    )
    
    model.add_reaction(
        name="v_PSIId",
        fn=_v_PSIId,
        args=[n.ps2("_damaged"), "PARa", "alphar", "Kinh", "Krep"],
        stoichiometry={
            n.ps2("_damaged"): 1
        },
        unit=units.per_second
    )
    
    model.add_reaction(
        name="v_fR",
        fn=inc_dec_rate,
        args=["fRss", "fR", "KiR", "KdR"],
        stoichiometry={
            "fR": 1
        },
        unit=units.per_second
    )
    
    model.add_reaction(
        name="v_gw",
        fn=_v_g,
        args=["Ci", n.co2("_cytosol"), "gw"],
        stoichiometry={
            n.co2("_cytosol"): Derived(fn=div, args=["Mv", "Vref"]),
            "Ci": Derived(fn=neg_div, args=["Mv", "Vref"])
        },
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_reaction(
        name="v_Rp",
        fn=_v_Rp,
        args=["PR", "kPR"],
        stoichiometry={
            n.co2("_cytosol"): Derived(fn=half_div, args=["Mv", "Vref"]),
            "PR": -1
        },
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_reaction(
        name="v_Rm",
        fn=simple_arrhenius,
        args=["Rm25", "DHaRm", "Tref", "R", "T"],
        stoichiometry={
            n.co2("_cytosol"): Derived(fn=div, args=["Mv", "Vref"])
        },
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_reaction(
        name="v_gv",
        fn=_v_g,
        args=[n.co2("_cytosol"), n.co2("_chloroplast"), "gc"],
        stoichiometry={
            n.co2("_cytosol"): Derived(fn=neg_div, args=["Mv", "Vref"]),
            n.co2("_chloroplast"): Derived(fn=div, args=["Mv", "Vref"])
        }
    )
    
    model.add_reaction(
        name="A",
        fn=_A,
        args=[n.co2("_chloroplast"), "Ca", "gsc", "gbc"],
        stoichiometry={
            "Ci": Derived(fn=div, args=["Mv", "Vref"])
        },
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_reaction(
        name="v_gsw",
        fn=inc_dec_rate,
        args=["gss", "gsw", "Kgsi", "Kgsd"],
        stoichiometry={
            "gsw": 1
        },
        unit=units.per_second
    )
    
    model.add_reaction(
        name="v_ca",
        fn=_v_ca,
        args=["Flow", "Ca", "CO2R", "leaf_surface", "A"],
        stoichiometry={
            "Ca": Derived(fn=_stoic_chamber, args=["R", "Ta", "volume_chamber", "air_pressure"])
        },
        unit=unit_mumol_per_sqm / units.second
    )
    
    model.add_reaction(
        name="v_H2OS",
        fn=_v_H2OS,
        args=["Flow", "H2OS", "H2OR", "leaf_surface", "transpiration"],
        stoichiometry={
            "H2OS": Derived(fn=_stoic_chamber, args=["R", "Ta", "volume_chamber", "air_pressure"])
        },
        unit=units.mmol / (units.mol * units.second)
    )
    
    model.add_reaction(
        name="v_fRB",
        fn=_v_fRB,
        args=["VrTPU", "VrJ", "phi", "Vc", "fRB", "fRuBP", "Flux0", "PAR", "fRBmin", "fRBmax", "fRBss_nr", "Krca", "RCA", "fRCA", "KdRB"],
        stoichiometry={
            "fRB": 1
        },
        unit=units.per_second
    )
    
    return model