# Parameters from generate_MiniModel_model()
MINI_MODEL_PARAMS = {
    "sigma2": 0.5, "Jmax25": 0.00013928, "DHaJmax": 36210000, "DHdJmax": 215900000, "DsJmax": 690000,
    "kD0": 4.55e+08, "kDinh": 5e+09, "kf": 5.6e+07, "kp": 2.654e+09, "fcyc": 0.1, "fpseudo": 0.1,
    "theta": 0.7, "gamma1": 0.2, "gamma2": 0.6, "gamma3": 0.2, "PhiqEmax": 0.2, "KiqEp": 0.0187,
    "KdqEp": 0.0239, "KiqEz": 0.00187, "KdqEz": 0.00239, "Kinh0": 0.1, "fprot": 0.1, "Krep25": 0.000192,
    "DHaKrep": 160800000, "DHdKrep": 233230000, "DsKrep": 780000, "alphar_alpha25": 6550,
    "DHaAlphar": 67320000, "DHaKalpha": 90500000, "DsKalpha": 1080000, "DHdKalpha": 3.28e+08,
    "Iac": 1.6e-06, "alpharac": 0.05, "alpharav": 0.25, "thetaalphar": 0.36, "Kialpha25": 0.00149,
    "Kdalpha25": 0.00186, "fR0": 0.04, "alphafR": 2500, "thetafR": 0.96, "KiR": 0.00628, "KdR": 0.0075,
    "Vrmax": 0.00011865, "KmPGA": 5e-06, "RB": 1.59e-05, "Kc25": 4.16, "DHaKc": 41820000, "Ko25": 1.26,
    "DHaKo": 55150000, "Kmc25": 0.0002617, "DHaKmc": 49430000, "Kmo25": 0.1985, "DHaKmo": 29080000,
    "KaRCA": 0.0102, "ac": 0.27, "bc": 14000, "KdRB": 0.00068, "Krca": 0.0863, "fRBmin": 0.48,
    "O2": 0.21, "RCA": 0.11737, "DHdRCA": 290200000, "ToRCA": 300.4, "DHaRCA": 3e+07, "KmRuBP": 0.02,
    "Vch": 1e-05, "KiPGA": 0.84, "TPU25": 7.47e-06, "DHaTPU": 57500000, "DHdTPU": 246700000,
    "DsTPU": 790000, "DHaGc": 70200000, "DHdGc": 9.4e+07, "DsGc": 320000, "DHaGw": 70200000,
    "DHdGw": 9.4e+07, "DsGw": 320000, "Rm25": 9.9e-07, "DHaRm": 56200000, "kPR": 0.024, "Vref": 0.000155,
    "Scm": 7.1, "Sm": 9.8, "falphaSc": 0.93, "gcm25": 0.39, "gw25": 0.75, "D0": 740000, "fI0": 0.39,
    "gswm": 0.48, "alphafI": 767, "thetafI": 0.88, "Kgsi": 0.00114, "Kgsd": 0.00114, "gbw": 9.2,
    "volume_chamber": 8e-05, "leaf_surface": 2e-04, "Flow": 5e-04, "alphab": 0.92, "alphag": 0.72,
    "alphared": 0.83, "alphabp": 0.66, "alphagp": 0.6, "alpharp": 0.8
}

# Convert dict to ordered list to match C++ params[i]
param_keys = list(MINI_MODEL_PARAMS.keys())
param_vector = [MINI_MODEL_PARAMS[k] for k in param_keys]