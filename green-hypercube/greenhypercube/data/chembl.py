"""ChEMBL adapter: an independent, measured bioactivity reward.

This is the leakage-resistant reward source. For each species we take the
chemicals attributed to it (from Dr. Duke's, used here only as a species ->
compound bridge) and look up *measured* assay potency in ChEMBL (pChEMBL values
from dose-response bioassays). A species' reward is an aggregate of the measured
potency of its compounds.

Why this is more defensible than use-based reward:
- Measured potency is determined by chemistry-vs-biological-target, not by
  whether the plant was historically adopted, so it does not encode the outcome
  of the very discovery process we are modeling.
- It lives on a different axis from the sensory CUE (which is the chemical
  *class*, e.g. bitter/aromatic), reducing same-record circularity.

Residual caveat (documented, not eliminated): the species->compound bridge still
comes from Dr. Duke's, and ChEMBL coverage is itself biased toward studied
compounds. ``requests`` is imported lazily.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..utils import get_logger

log = get_logger("data.chembl")

CHEMBL_MOLECULE = "https://www.ebi.ac.uk/chembl/api/data/molecule"
CHEMBL_MOLECULE_SEARCH = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
CHEMBL_ACTIVITY = "https://www.ebi.ac.uk/chembl/api/data/activity"


def _robust_session():
    """A requests session with retry/backoff (ChEMBL 5xx are intermittent)."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    sess = requests.Session()
    retry = Retry(
        total=2, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",), raise_on_status=False,
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.headers.update({"Accept": "application/json", "User-Agent": "greenhypercube"})
    return sess


def _chembl_available(sess) -> bool:
    """Single health probe so a transient outage fails fast (no retry storm)."""
    try:
        r = sess.get(CHEMBL_MOLECULE_SEARCH, params={"q": "aspirin", "format": "json"}, timeout=30)
        return r.status_code == 200
    except Exception:  # pragma: no cover - network dependent
        return False


def _activities_max_pchembl(sess, chembl_id: str) -> float:
    """Best (max) measured pChEMBL value across a molecule's bioassays, or 0."""
    a = sess.get(
        CHEMBL_ACTIVITY,
        params={"molecule_chembl_id": chembl_id, "pchembl_value__isnull": "false",
                "format": "json", "limit": 50},
        timeout=45,
    )
    a.raise_for_status()
    pvals = [float(act["pchembl_value"]) for act in a.json().get("activities", [])
             if act.get("pchembl_value")]
    return float(np.max(pvals)) if pvals else 0.0


def _normalize_potency(raw: np.ndarray) -> np.ndarray:
    """Map measured pChEMBL (~4..10) to [0,1]; species with no data stay 0."""
    if raw.max() > 0:
        assay = np.clip((raw - 4.0) / 6.0, 0.0, 1.0)
        assay[raw == 0] = 0.0
        return assay
    return raw


def fetch_bioassay_via_compounds(
    species: pd.DataFrame, compounds: pd.DataFrame, cfg: DataConfig, max_compounds: int = 8
) -> pd.DataFrame:
    """Per-species reward from ChEMBL potency of LOTUS-bridged compounds.

    ``compounds`` has columns species_id, inchikey, compound (label). Each
    compound is resolved in ChEMBL by InChIKey (exact) with a name-search
    fallback; a species' reward is the best measured potency among its compounds,
    normalized to [0, 1]. Species with no resolvable measured activity get 0 --
    the sparse floor that makes an unfiltered flora a real haystack.
    """
    sess = _robust_session()
    if not _chembl_available(sess):
        log.warning("ChEMBL API unavailable; reward left empty -- rerun the reward "
                    "step (build --force) once the service recovers")
        return pd.DataFrame({"species_id": species["species_id"].values,
                             "assay_value": np.zeros(len(species), dtype=float)})
    cache: dict[str, float] = {}

    # Outcome of a single lookup: (potency, ok). ok=False means the *service*
    # failed (not merely "no measured data"), which feeds the circuit breaker.
    def potency_inchikey(ik: str) -> tuple[float, bool]:
        try:
            r = sess.get(f"{CHEMBL_MOLECULE}/{ik}.json", timeout=45)
            if r.status_code == 404:
                return 0.0, True  # resolved: this structure simply isn't in ChEMBL
            if r.status_code != 200:
                return 0.0, False  # transient service error
            cid = r.json().get("molecule_chembl_id")
            return (_activities_max_pchembl(sess, cid), True) if cid else (0.0, True)
        except Exception:  # pragma: no cover - network dependent
            return 0.0, False

    def potency_name(name: str) -> tuple[float, bool]:
        if not name:
            return 0.0, True
        q = name.replace("-", " ").replace("_", " ").strip().lower()
        try:
            r = sess.get(CHEMBL_MOLECULE_SEARCH, params={"q": q, "format": "json"}, timeout=45)
            if r.status_code != 200:
                return 0.0, False
            mols = r.json().get("molecules", [])
            if mols:
                return _activities_max_pchembl(sess, mols[0]["molecule_chembl_id"]), True
            return 0.0, True
        except Exception:  # pragma: no cover - network dependent
            return 0.0, False

    def potency(ik: str, name: str) -> tuple[float, bool]:
        key = ik or name
        if key in cache:
            return cache[key], True
        val, ok = potency_inchikey(ik) if ik else (0.0, False)
        if not ok or val == 0.0:  # try name fallback for misses/failures
            v2, ok2 = potency_name(name)
            val, ok = (max(val, v2), ok or ok2)
        if ok:
            cache[key] = val
        time.sleep(0.03)
        return val, ok

    raw = np.zeros(len(species), dtype=float)
    sid_to_row = {int(s): i for i, s in enumerate(species["species_id"])}
    consecutive_fail = 0
    aborted = False
    if len(compounds):
        for sid, grp in compounds.groupby("species_id"):
            if aborted:
                break
            row = sid_to_row.get(int(sid))
            if row is None:
                continue
            best = 0.0
            for _, c in grp.head(max_compounds).iterrows():
                val, ok = potency(str(c.get("inchikey", "")), str(c.get("compound", "")))
                consecutive_fail = 0 if ok else consecutive_fail + 1
                if consecutive_fail >= 12:  # circuit breaker: ChEMBL degraded mid-run
                    log.warning("ChEMBL degraded mid-run (%d consecutive failures); "
                                "stopping early -- rerun reward once it recovers",
                                consecutive_fail)
                    aborted = True
                    break
                best = max(best, val)
            raw[row] = best

    assay = _normalize_potency(raw)
    out = pd.DataFrame({"species_id": species["species_id"].values, "assay_value": assay})
    log.info("ChEMBL (via LOTUS): %d/%d species with measured potency",
             int((assay > 0).sum()), len(species))
    return out


def fetch_bioassay(
    species: pd.DataFrame, chemicals: pd.DataFrame, cfg: DataConfig, max_compounds: int = 6
) -> pd.DataFrame:
    """Aggregate measured ChEMBL potency to a per-species reward in [0, 1].

    Returns the normalized ``bioassay`` table (species_id, assay_value). Species
    with no resolvable measured activity get 0 (the sparse, mostly-empty floor).
    """
    sess = _robust_session()
    headers = {"Accept": "application/json"}
    if not _chembl_available(sess):
        log.warning("ChEMBL API unavailable; reward left empty -- rerun the reward "
                    "step (build --force) once the service recovers")
        return pd.DataFrame({"species_id": species["species_id"].values,
                             "assay_value": np.zeros(len(species), dtype=float)})

    # Cache compound -> best pChEMBL so repeated chemicals aren't re-queried.
    compound_potency: dict[str, float] = {}

    def potency_for(name: str) -> float:
        if name in compound_potency:
            return compound_potency[name]
        val = 0.0
        # Mirror compound names are uppercased/hyphenated (e.g. MASLINIC-ACID);
        # ChEMBL name search prefers a cleaner query.
        query = name.replace("-", " ").replace("_", " ").strip().lower()
        try:
            r = sess.get(
                CHEMBL_MOLECULE_SEARCH, params={"q": query, "format": "json"},
                headers=headers, timeout=45,
            )
            r.raise_for_status()
            mols = r.json().get("molecules", [])
            if mols:
                chembl_id = mols[0]["molecule_chembl_id"]
                a = sess.get(
                    CHEMBL_ACTIVITY,
                    params={"molecule_chembl_id": chembl_id, "format": "json", "limit": 50},
                    headers=headers, timeout=45,
                )
                a.raise_for_status()
                pvals = [
                    float(act["pchembl_value"])
                    for act in a.json().get("activities", [])
                    if act.get("pchembl_value")
                ]
                if pvals:
                    val = float(np.max(pvals))  # best measured potency
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("ChEMBL lookup failed for %r: %s", name, exc)
        compound_potency[name] = val
        time.sleep(0.05)  # be polite to the public API
        return val

    raw = np.zeros(len(species), dtype=float)
    sid_to_row = {int(s): i for i, s in enumerate(species["species_id"])}
    if len(chemicals):
        for sid, grp in chemicals.groupby("species_id"):
            row = sid_to_row.get(int(sid))
            if row is None:
                continue
            names = list(dict.fromkeys(grp["chemical"].astype(str)))[:max_compounds]
            potencies = [potency_for(n) for n in names]
            if potencies:
                raw[row] = float(np.max(potencies))  # species potency = best compound

    # Normalize measured pChEMBL (~4..10) to [0,1]; species with no data stay 0.
    if raw.max() > 0:
        assay = np.clip((raw - 4.0) / 6.0, 0.0, 1.0)
        assay[raw == 0] = 0.0
    else:
        assay = raw
    out = pd.DataFrame({"species_id": species["species_id"].values, "assay_value": assay})
    log.info("ChEMBL bioassay: %d/%d species with measured potency",
             int((assay > 0).sum()), len(species))
    return out
