# A4 solver sensitivity archive

Consolidated from stage0 (see stage0_log.txt for full detail).

- MP(c) recovery, interior max relative error: 0.760%
- Truncation K=20: mass 0.00e+00, 2nd tail moment 5.11e-15
- Truncation K=60: mass 0.00e+00, 2nd tail moment -5.55e-17
- Density difference K20 vs K60: 2.11e-15
- h sensitivity (h=1e-3 vs 5e-4), max |Δf| at left edge: 1.438e-01 (edge sharpening; report only)
- Dual-solver density deviation (x>0.15): 0.000%
- Zero-mass checks: all within 4e-8 of analytic formula.
- Companion route: auxiliary u = c·m, e = u + (c-1)/z; internal identity ∫1/(1+tu) = -z·e (see NOTE_伴随方程形式问题.md).
