# Spectral Costs of Multinomial and Residual Resampling in High-Dimensional Particle Ensembles

Code, data and figure sources for the paper:

> **Spectral Costs of Multinomial and Residual Resampling in High-Dimensional Particle Ensembles: A Free-Probability Analysis**
> Deming Song and Chi Kin Lam, Faculty of Applied Sciences, Macao Polytechnic University.
> Submitted to *IEEE Transactions on Signal Processing*.

This repository is the archival snapshot accompanying the manuscript. It contains the
numerical experiments referenced in the main text and the supplementary material, the
figure-generation scripts, the compact result tables, and the LaTeX sources of the
submission.

**Archived version:** `v1.0-submission` — see `CITATION.cff` for the Zenodo DOI once minted.

---

## 1. What is here

```
paper/
  MainText.pdf            submitted manuscript (12 pp., double column)
  supplementary.pdf       submitted supplementary material (proofs + tables)
  src/                    LaTeX sources of both
    MainText.tex supplement.tex proofs.tex dm_refs.bib IEEEtran.cls
    fig/                  10 vector PDFs as included by the submission

experiments/
  common/code/            shared solvers and plotting
    stage0_solvers.py       two independent spectral solvers (Definition 1)
    plot_panels.py          per-panel PDF renderer
  common/logs/            stage0 self-check output, replay verification report
  E1_count_law_spectrum/  code/ data/ fig/ logs/
  E2_rank_threshold/      code/ data/ fig/ logs/
  E3_same_ESS/            code/ data/ fig/ logs/
  E4_integer_boundary/    code/ data/ fig/ logs/
  T1_moment_identity/     code/ data/ logs/
  A1-A4_appendix/         code/ data/ fig/ logs/

docs/
  实验总指导_谱代价论文.md      experiment programme (zh)
  实验完成报告.md               completion report, all pass/fail criteria (zh)
  数据文件索引.md               data-file index, column-by-column (zh)
  论文写作快速参考.md           writing reference (zh)
  NOTE_伴随方程形式问题.md      solver note: two literature companion forms refuted (zh)
  PANELS_README.md              per-panel figure conventions and the 20-panel list (zh)
  ARCHIVE_NOTE.md               warning about superseded material (zh)
```

---

## 2. Experiment inventory and where each item appears in the paper

| Directory | Experiment | Main text | Supplement |
|---|---|---|---|
| `common/logs/stage0_*` | Solver unit tests (two independent implementations) | — | S23 |
| `T1_moment_identity` | Finite-sample moment identity, Prop. 4 / Cor. 19 | §IX.G | — |
| `E1_count_law_spectrum` | Count law → spectrum map | Fig. 1; Table I | S25 |
| `E2_rank_threshold` | Rank threshold and lower edge, 12-point sweep | — | S24 |
| `E3_same_ESS` | Equal ESS, different null spaces | Fig. 2; Table II | — |
| `E4_integer_boundary` | Residual randomization at the integer boundary | Fig. 3; Table III | S26 |
| `E4_integer_boundary` (spectral panels) | The residual atom-free window | Fig. 4 | — |
| `A1-A4_appendix` | Scale checks, non-Gaussian control, two-point population spectrum, solver sensitivity | — | Fig. S1; S27 |

---

## 3. Directory naming does **not** follow the paper's figure numbers

> **Read this before citing a file by path.**

The directories retain the experiment identifiers used during development (E1–E4, T1, A1–A4).
Those identifiers are **not** the figure numbers of the final manuscript. Two items moved
between the last draft and the submitted version:

| Experiment | In the final manuscript |
|---|---|
| `E2_rank_threshold` | **no figure.** E2 survives only as two tables in supplementary S24. |
| `E3_same_ESS` | **Fig. 2** (was Fig. 3 in draft numbering) |
| `E4_integer_boundary` | **Fig. 3 and Fig. 4** (split into counts and spectrum) |
| `T1_moment_identity` | **no table.** T1 survives as narrative in §IX.G. |

An earlier directory layout used the draft figure numbers (`Fig1_E1`, `Fig2_E2`,
`Fig3_E3`, `Fig4_E4`, `Tab4_T1`). That layout is **not** reproduced here, because under it
`Fig2_E2` would name a figure that the paper does not contain.

---

## 4. Panel-level usage of the 20 figure panels

Each experiment writes its panels into four width presets (see `docs/PANELS_README.md`):
`w1.66_col2`, `w2.33_span3`, `w3.40_single`, `w3.50_span2`. Every preset is included, so
that the archive is complete and the choice can be re-made.

**Which preset the submission actually used:**

| Panel | Used in | Preset used |
|---|---|---|
| `E1_twopoint_spectrum_{pre,multinomial,residual}` | main text Fig. 1 | `w2.33_span3` |
| `E3_zero_atom_bars` | main text Fig. 2 | `w3.50_span2` |
| `E4_counts_{dev2,RnN}` | main text Fig. 3 | `w3.50_span2` |
| `E4_spectral_{pre,multinomial,residual}` | main text Fig. 4 | `w2.33_span3` |
| `A3_population_spectrum` | supplement Fig. S1 | `w3.40_single` |

**Panels produced but not used in the submitted manuscript (10 of 20).** They are retained
for completeness of the record — they were pre-registered outputs, not discarded errors:

| Panel(s) | Why unused |
|---|---|
| `E2_partA_zero_mass`, `E2_partB_lower_edge` | E2's figure was cut; the numbers appear in S24 tables |
| `E1_uniform_spectrum_{pre,multinomial,residual}` | the uniform profile is degenerate; text reports it via Cor. `coro:uniform-contrast` instead |
| `E1_{uniform,twopoint}_count_{multinomial,residual}` (4) | count PMFs moved to supplement S25 as tables |
| `E3_cdf` | replaced by `E3_zero_atom_bars` in Fig. 2 |

`w1.66_col2` is documented in `PANELS_README.md` as **not recommended** (legend area
crowds the spectra); it is included only for completeness.

---

## 5. Reproducing

Requires Python 3 with `numpy`, `scipy`, `matplotlib`.

```bash
cd experiments/E1_count_law_spectrum/code
python stage2_E1.py            # writes ./output/  (E1_moments.csv etc.)
```

Each stage script writes to an `output/` directory beside itself and is seeded
(`SEED = 20260915`) for bit-reproducibility. `stage3_appendix.py` and `stage2_E1.py`
import `stage0_solvers.py`, which lives in `experiments/common/code/`; put that directory
on `PYTHONPATH`, or copy the script next to the one you are running.

To regenerate the per-panel PDFs at a chosen width:

```bash
cd experiments/common/code
python plot_panels.py --width span3      # or single / span2 / col2, or any numeric width
```

`plot_panels.py` does not recompute the statistics in the CSVs; for panels whose plotting
arrays were never written to disk it replays the original seed and asserts the replayed
statistics against the stored CSVs (393 items, maximum absolute deviation 4.441e-16 — see
`experiments/common/logs/REPLAY_VERIFICATION.txt`).

---

## 6. Data provenance

Every CSV under `experiments/*/data/` is **byte-identical** (md5) to the corresponding file
in the paper's working `experiments/output/` directory. The supplementary PDFs were built
from these same files. No value in this archive has been recomputed, rounded, or edited for
release.

The LaTeX in `paper/src/` is exactly the source of `paper/MainText.pdf` and
`paper/supplementary.pdf`; `paper/src/fig/` holds the 10 vector PDFs as included by the
submission.

---

## 7. Known defect, recorded

`E4_integer_boundary` originally produced a combined `E4_spectral.png`. Its **pre-resampling
panel draws a flat line at zero** instead of the MP(c) prediction, because
`stage1_E4.py:157` calls `density_grid` with a 61-element `supp` against a 1-element `probs`
and the returned density is identically zero. `stage1_E4.py` was **not** modified.

The corrected theory curves are in the per-panel PDFs, produced by `plot_panels.py`, which
fixes the call (see the note at `plot_panels.py:456`). The affected multi-panel PNGs
(`E4_spectral.png`, and the superseded `E1_figure_*.png`, `E2_figure.png`, `E3_figure.png`,
`E4_figure.png`, `A3_population.png`) are **excluded from this archive**; the per-panel PDFs
are authoritative. All 393 replay assertions still pass with the fix.

A second solver-level finding is recorded in `docs/NOTE_伴随方程形式问题.md`: of the two
companion-side Stieltjes equations common in the literature, neither holds under this
paper's parametrisation. The correct route is the algebraic rewrite used in
`stage0_solvers.py` and documented in supplementary S23.

---

## 8. Licence

- Code (`*.py`): MIT — see `LICENSE`.
- Data, figures and documentation: CC BY 4.0 — see `LICENSE-data`.
- `paper/src/IEEEtran.cls` is a third-party file from IEEE, distributed under the LaTeX
  Project Public License; it is included unchanged so that the sources compile. It is not
  covered by the licences above.

## 9. Citation

See `CITATION.cff`. Please cite the paper if you use this code or these data.
