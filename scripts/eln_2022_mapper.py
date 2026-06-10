"""Deterministic ELN 2022 risk mapper.

Implements the rule-based mapping from karyotype + targeted gene panel to the
three ELN-2022 risk strata (favorable / intermediate / adverse) per

    Döhner H, Wei AH, Appelbaum FR, et al. (2022). Diagnosis and management of
    AML in adults: 2022 recommendations from an international expert panel on
    behalf of the ELN. Blood, 140(12):1345-1377. DOI 10.1182/blood.2022016867.

Lock the version-tag in `MAPPER_VERSION` BEFORE any external-cohort evaluation;
a hash of this file's content goes into the W&B run config so any later edit
forces a re-evaluation (per TRIPOD+AI item 9 deviation reporting).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

MAPPER_VERSION = "ELN-2022-mapper-v1.0.0"


class Risk(str, Enum):
    FAVORABLE = "favorable"
    INTERMEDIATE = "intermediate"
    ADVERSE = "adverse"


# ELN 2022 myelodysplasia-related (MR) gene set (Table 5 of Döhner 2022).
MR_GENES: frozenset[str] = frozenset({
    "ASXL1", "BCOR", "EZH2", "RUNX1", "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2",
})

# ELN 2022 myelodysplasia-related cytogenetic abnormalities (Table 5).
MR_CYTOGENETICS: frozenset[str] = frozenset({
    "complex_karyotype",        # ≥3 unrelated abnormalities, no recurring genetic abnormality
    "monosomal_karyotype",
    "del(5q)", "del(5)(q)", "-5",
    "-7", "del(7q)",
    "-17", "del(17p)", "abn(17p)",
    "del(12p)", "t(11;16)",
    "i(17q)", "t(3;5)",
    "t(2;11)", "t(5;12)",
})

FAVORABLE_RECURRING: frozenset[str] = frozenset({
    "t(8;21)(q22;q22)", "RUNX1-RUNX1T1",
    "inv(16)(p13.1q22)", "t(16;16)(p13.1;q22)", "CBFB-MYH11",
})

ADVERSE_RECURRING: frozenset[str] = frozenset({
    "t(6;9)(p22.3;q34.1)", "DEK-NUP214",
    "t(v;11q23.3)", "KMT2A-rearranged_non_PTD",
    "t(9;22)(q34.1;q11.2)", "BCR-ABL1",
    "inv(3)(q21.3q26.2)", "t(3;3)(q21.3;q26.2)", "GATA2-MECOM(EVI1)",
    "t(8;16)(p11.2;p13.3)", "KAT6A-CREBBP",
})


@dataclass
class PatientGenetics:
    """Inputs to ELN-2022 classification for a single patient.

    All gene-mutation flags are True iff the mutation is *pathogenic* per the
    cohort's variant-classification pipeline (e.g., OncoKB, ClinVar pathogenic +
    OncoKB oncogenic, or Bottomly 2022 curated calls).
    """
    # Recurring genetic abnormalities (cytogenetics)
    cytogenetic_findings: frozenset[str] = field(default_factory=frozenset)

    # Single-gene mutation calls
    npm1_mut: bool = False
    flt3_itd: bool = False                # ITD = internal tandem duplication
    flt3_itd_allelic_ratio: float | None = None     # ≥0.5 was adverse in ELN 2017; removed in 2022
    cebpa_bzip_inframe: bool = False      # ELN 2022 changed CEBPA criterion
    tp53_mut: bool = False                # any pathogenic TP53 mutation
    asxl1_mut: bool = False
    bcor_mut: bool = False
    ezh2_mut: bool = False
    runx1_mut: bool = False
    sf3b1_mut: bool = False
    srsf2_mut: bool = False
    stag2_mut: bool = False
    u2af1_mut: bool = False
    zrsr2_mut: bool = False

    # Provenance — for the mapper audit trail
    sample_id: str = ""
    cohort: str = ""

    def mr_gene_mutations(self) -> frozenset[str]:
        flags = {
            "ASXL1": self.asxl1_mut, "BCOR": self.bcor_mut, "EZH2": self.ezh2_mut,
            "RUNX1": self.runx1_mut, "SF3B1": self.sf3b1_mut, "SRSF2": self.srsf2_mut,
            "STAG2": self.stag2_mut, "U2AF1": self.u2af1_mut, "ZRSR2": self.zrsr2_mut,
        }
        return frozenset(g for g, has in flags.items() if has)

    def has_mr_cytogenetics(self) -> bool:
        return bool(self.cytogenetic_findings & MR_CYTOGENETICS)

    def has_favorable_recurring(self) -> bool:
        return bool(self.cytogenetic_findings & FAVORABLE_RECURRING)

    def has_adverse_recurring(self) -> bool:
        return bool(self.cytogenetic_findings & ADVERSE_RECURRING)


def classify(p: PatientGenetics) -> tuple[Risk, list[str]]:
    """Return (risk, rationale) where rationale is the audit trail of rules fired.

    The order of rules matches Döhner 2022 Table 5:
        1. ADVERSE if: TP53-mut, or MR-gene mutation, or MR cytogenetics, or
           adverse recurring abnormality.  These OVERRIDE favorable defining
           lesions per the 2022 update.
        2. FAVORABLE if: t(8;21), inv(16)/t(16;16), or NPM1-mut (without
           FLT3-ITD), or in-frame bZIP CEBPA.
        3. INTERMEDIATE otherwise.

    Note: ELN 2022 removed the FLT3-ITD allelic-ratio sub-classification —
    NPM1-mut + FLT3-ITD is now intermediate regardless of ratio.
    """
    rationale: list[str] = []

    # --- Adverse-overriding lesions (Döhner 2022 Table 5 footnote a) -------
    if p.tp53_mut:
        rationale.append("TP53 mutation -> adverse")
        return Risk.ADVERSE, rationale

    mr_mut = p.mr_gene_mutations()
    if mr_mut:
        rationale.append(f"MR-gene mutation(s): {sorted(mr_mut)} -> adverse")
        return Risk.ADVERSE, rationale

    if p.has_mr_cytogenetics():
        hits = sorted(p.cytogenetic_findings & MR_CYTOGENETICS)
        rationale.append(f"MR-related cytogenetics: {hits} -> adverse")
        return Risk.ADVERSE, rationale

    if p.has_adverse_recurring():
        hits = sorted(p.cytogenetic_findings & ADVERSE_RECURRING)
        rationale.append(f"adverse-defining recurring abnormality: {hits} -> adverse")
        return Risk.ADVERSE, rationale

    # --- Favorable lesions ---------------------------------------------------
    if p.has_favorable_recurring():
        hits = sorted(p.cytogenetic_findings & FAVORABLE_RECURRING)
        rationale.append(f"favorable recurring abnormality: {hits} -> favorable")
        return Risk.FAVORABLE, rationale

    if p.npm1_mut and not p.flt3_itd:
        rationale.append("NPM1-mut without FLT3-ITD -> favorable")
        return Risk.FAVORABLE, rationale

    if p.cebpa_bzip_inframe:
        rationale.append("in-frame bZIP CEBPA -> favorable (ELN 2022 update)")
        return Risk.FAVORABLE, rationale

    # --- Intermediate fall-through ------------------------------------------
    if p.npm1_mut and p.flt3_itd:
        rationale.append("NPM1-mut + FLT3-ITD -> intermediate (ELN 2022; allelic ratio not used)")
        return Risk.INTERMEDIATE, rationale

    if p.flt3_itd and not p.npm1_mut:
        rationale.append("FLT3-ITD without NPM1-mut -> intermediate")
        return Risk.INTERMEDIATE, rationale

    rationale.append("no defining lesion -> intermediate")
    return Risk.INTERMEDIATE, rationale


def mapper_content_hash() -> str:
    """SHA-256 of this file's text, locked into experiment configs."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def classify_batch(patients: Iterable[PatientGenetics]) -> list[dict]:
    """Apply `classify` to a batch and return list-of-dicts ready for parquet."""
    rows: list[dict] = []
    for p in patients:
        risk, why = classify(p)
        rows.append({
            "sample_id": p.sample_id,
            "cohort": p.cohort,
            "eln_2022_risk": risk.value,
            "mapper_version": MAPPER_VERSION,
            "mapper_hash": mapper_content_hash()[:16],
            "rationale": "; ".join(why),
        })
    return rows


if __name__ == "__main__":
    # Smoke test against three textbook cases.
    cases = [
        PatientGenetics(sample_id="ex1", cohort="ex",
                        cytogenetic_findings=frozenset({"t(8;21)(q22;q22)"})),
        PatientGenetics(sample_id="ex2", cohort="ex",
                        npm1_mut=True, flt3_itd=False),
        PatientGenetics(sample_id="ex3", cohort="ex",
                        tp53_mut=True, cytogenetic_findings=frozenset({"complex_karyotype"})),
        PatientGenetics(sample_id="ex4", cohort="ex",
                        cebpa_bzip_inframe=True),
        PatientGenetics(sample_id="ex5", cohort="ex",
                        asxl1_mut=True),
    ]
    for c in cases:
        risk, why = classify(c)
        print(f"  {c.sample_id}: {risk.value:13s} <- {'; '.join(why)}")
    print(f"\nmapper_version={MAPPER_VERSION}\nmapper_hash={mapper_content_hash()[:16]}…")
