import numpy as np

from casmi26.adducts import precursor_to_neutral
from casmi26.metric import format_smiles_field, mrr_at_k
from casmi26.similarity import entropy_similarity, modified_cosine
from casmi26.spectra import clean_peaks


def test_identical_spectra_cosine_is_one():
    mz = np.array([70.0, 100.0, 150.0])
    inten = np.array([0.2, 1.0, 0.5])
    score = modified_cosine(mz, inten, mz, inten, 200.0, 200.0, tol=0.02)
    assert score > 0.99


def test_shifted_peaks_still_match():
    mz_a = np.array([80.0, 120.0, 180.0])
    mz_b = mz_a + 17.0
    inten = np.array([0.4, 1.0, 0.6])
    score = modified_cosine(mz_a, inten, mz_b, inten, 200.0, 217.0, tol=0.05)
    assert score > 0.9


def test_entropy_self_similarity():
    mz = np.array([50.0, 90.0, 130.0])
    inten = np.array([0.3, 1.0, 0.4])
    assert entropy_similarity(mz, inten, mz, inten) > 0.99


def test_clean_drops_above_precursor():
    mz = np.array([50.0, 90.0, 400.0])
    inten = np.array([1.0, 0.5, 0.8])
    out_mz, _ = clean_peaks(mz, inten, precursor_mz=200.0, intensity_floor=0.01, min_peaks=1)
    assert np.all(out_mz <= 202.0)


def test_neutral_mass_mh():
    # caffeine [M+H]+ ~ 195.088
    m = precursor_to_neutral(195.0876, "[M+H]+")
    assert abs(m - 194.0803) < 0.01


def test_format_and_mrr():
    field = format_smiles_field(["CCO", "CCO", "OCC(O)CO"])
    assert field == "CCO;OCC(O)CO"
    score = mrr_at_k([["C", "CCO"]], ["CCO"], k=25)
    assert abs(score - 0.5) < 1e-6
