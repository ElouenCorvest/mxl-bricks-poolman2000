from __future__ import annotations

from typing import Literal

from mxlbricks.names import loc

EMPTY: Literal[""] = ""

def add_temp(name: str, temp: EMPTY) -> str:
    return name + temp

def energy_activation(name: EMPTY) -> str:
    """Activation Energy"""
    return "Energy_activation" + name

def k(enzyme: str, substrate: EMPTY) -> str:
    if substrate is EMPTY:
        return f"kcat_{enzyme}"
    return f"k_{enzyme}_{substrate}"

def entropy(name: EMPTY) -> str:
    """Entropy"""
    return "Entropy" + name

def energy_deactivation(name: EMPTY) -> str:
    """Deactivation Energy"""
    return "Energy_deactivation" + name