"""Adduct → neutral-mass conversion for the CASMI 2026 ion types."""

from __future__ import annotations

PROTON = 1.007276466812
WATER = 18.010564684
NH3 = 17.026549101
NA = 22.989269282
K = 38.9631579
CL35 = 34.969402203
HCOOH = 46.0054793034

# delta such that:  neutral_mass = precursor_mz * abs(charge) - delta
ADDUCTS: dict[str, dict[str, float | int]] = {
    "[M+H]+": {"delta": PROTON, "charge": 1},
    "[M+NH4]+": {"delta": NH3 + PROTON, "charge": 1},
    "[M-H2O+H]+": {"delta": PROTON - WATER, "charge": 1},
    "[M-2H2O+H]+": {"delta": PROTON - 2.0 * WATER, "charge": 1},
    "[M+Na]+": {"delta": NA, "charge": 1},
    "[M+K]+": {"delta": K, "charge": 1},
    "[M-H]-": {"delta": -PROTON, "charge": -1},
    "[M-H2O-H]-": {"delta": -WATER - PROTON, "charge": -1},
    "[M+CH2O2-H]-": {"delta": HCOOH - PROTON, "charge": -1},
    "[M+Cl]-": {"delta": CL35, "charge": -1},
    "[M+H-H2O]+": {"delta": PROTON - WATER, "charge": 1},
    "[M+H-2H2O]+": {"delta": PROTON - 2.0 * WATER, "charge": 1},
    "[M+HCOO]-": {"delta": HCOOH - PROTON, "charge": -1},
}


def adduct_delta(adduct: str) -> float:
    info = ADDUCTS.get(str(adduct).strip())
    if info is None:
        return PROTON
    return float(info["delta"])


def adduct_charge(adduct: str) -> int:
    info = ADDUCTS.get(str(adduct).strip())
    if info is None:
        return 1 if "+" in str(adduct) else -1
    return int(info["charge"])


def precursor_to_neutral(precursor_mz: float, adduct: str) -> float:
    z = abs(adduct_charge(adduct)) or 1
    return float(precursor_mz) * z - adduct_delta(adduct)


def ppm_delta(mass: float, ppm: float) -> float:
    return abs(mass) * ppm * 1e-6


def within_ppm(a: float, b: float, ppm: float) -> bool:
    return abs(a - b) <= ppm_delta((a + b) * 0.5, ppm)
