"""Local MRR@25 using InChIKey14 after tautomer canonicalization."""

from __future__ import annotations


def inchikey14(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        return smiles.strip()[:14] if smiles else None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)
    except Exception:
        pass
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not key:
        return None
    return key.split("-")[0]


def mrr_at_k(predictions: list[list[str]], labels: list[str], k: int = 25) -> float:
    if not predictions:
        return 0.0
    total = 0.0
    for preds, gold in zip(predictions, labels):
        gold_key = inchikey14(gold)
        if not gold_key:
            continue
        rank = 0
        for i, smi in enumerate(preds[:k], start=1):
            if inchikey14(smi) == gold_key:
                rank = i
                break
        total += (1.0 / rank) if rank else 0.0
    return total / len(predictions)


def format_smiles_field(smiles: list[str]) -> str:
    cleaned = []
    seen = set()
    for smi in smiles:
        s = str(smi).strip()
        if not s or s in seen:
            continue
        if "," in s or "\n" in s:
            continue
        seen.add(s)
        cleaned.append(s)
        if len(cleaned) == 25:
            break
    if not cleaned:
        raise ValueError("a molecule must have at least one SMILES guess")
    return ";".join(cleaned)
