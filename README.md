# Tau Core Paper 25 — Charged-Fermion Mass Hierarchies, Koide Equipartition, and Zero-Parameter CKM Closure

**Title:** *Charged-Fermion Mass Hierarchies, Koide Equipartition, and Zero-Parameter CKM Closure from a Tetrahedral $\mathrm{Spin}(10) \to \mathrm{Pati\text{--}Salam} \to \mathrm{SM}$ Renormalization Tower*  
**Author:** Jozsef Olcsak (ORCID: `0009-0003-7311-6522`)  
**Target Journals:** *Physical Review D* / *JHEP*

## One-Command Reproducibility

```bash
python3 scripts/verify_paper25_fermion_masses_ckm_rg.py
python3 scripts/generate_paper25_figures.py
```

## Contents
- `paper25_submission_source/main.tex` — Self-contained English journal manuscript (*PRD / JHEP* format)
- `paper25_hungarian_source/main_hu.tex` — Complete Hungarian companion edition
- `scripts/verify_paper25_fermion_masses_ckm_rg.py` — Exact `SymPy` proofs + 2-loop SM + Pati–Salam RGEs + Takagi optimization + Look-Elsewhere audit
- `scripts/generate_paper25_figures.py` — Generates all 4 vector PDF figures in `figures/`
- `scripts/paper25_verification_results.json` — Machine-verified numerical ledger
