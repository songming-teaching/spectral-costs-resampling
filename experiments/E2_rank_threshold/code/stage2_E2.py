# stage2_E2.py
# E2 — Rank threshold and lower edge (Proposition 8, Theorem 12, Proposition 13).
# Uniform weights, multinomial resampling. p = 1 − e^{−1} = 0.632121.
# Part A (count-only): q_{0,N} = (d − M_N)_+/d, N=10^4, 300 reps — no eigendecomposition.
# Part B: λ_min⁺ via smaller Gram (or SVD), N=1000, 15 reps per c.
# Legend levels: universal lower bound |√c−√p|² / true lower edge (Prop 13, 1-D root) /
#                finite-sample λ_min⁺ (median & quartiles).
# Outputs: output/E2_atom.csv, E2_edge.csv, E2_figure.png, E2_log.txt

import numpy as np, os, csv
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260915
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

p = 1.0 - np.exp(-1.0)          # surviving fraction, uniform multinomial
Cs = [0.35, 0.45, 0.55, 0.60, 0.6321, 0.66, 0.72, 0.80, 0.90, 1.00, 1.08, 1.15]

log = []
def note(s):
    print(s); log.append(s)

# ---------------- Part A: count-only zero mass ----------------
note("=== Part A: count-only, N=10^4, 300 reps ===")
N_A, R_A = 10_000, 300
atom_rows = []
for c in Cs:
    dA = int(round(c * N_A))
    q = np.empty(R_A)
    w = np.full(N_A, 1.0 / N_A)
    for r in range(R_A):
        n = rng.multinomial(N_A, w)
        M = np.count_nonzero(n)
        q[r] = max(dA - M, 0) / dA
    th = max(0.0, 1.0 - (1.0 - np.exp(-1.0)) / c)
    atom_rows.append(dict(c=c, q0_emp=q.mean(), q0_se=q.std(ddof=1) / np.sqrt(R_A), q0_th=th))
    note(f"c={c:5.3f}  q0 {q.mean():.4f}±{q.std(ddof=1)/np.sqrt(R_A):.4f}  th {th:.4f}")

# ---------------- Prop 13: true lower edge via critical equation ----------------
def z_of(u, y, Kmax=200):
    ks = np.arange(Kmax + 1)
    logp = -1.0 + ks * np.log(1.0) * 0  # Poi(1): p_k = e^{-1}/k!
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, Kmax + 1)))])
    pk = np.exp(-1.0 - logfact)
    return -1.0 / u + y * np.sum(pk * ks / (1.0 + ks * u))

def zp_of(u, y, Kmax=200):
    ks = np.arange(Kmax + 1)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, Kmax + 1)))])
    pk = np.exp(-1.0 - logfact)
    return 1.0 / u ** 2 - y * np.sum(pk * ks ** 2 / (1.0 + ks * u) ** 2)

def true_lower_edge(c):
    """Proposition 13: lower edge of the positive part.
    Branch rule (derived and empirically validated in stage0 development):
    - subcritical c<p: max of z on the (0,∞) branch (exists iff c<p);
    - supercritical c>p: max of z on the (−∞,−1) branch (exists iff c>p);
    - critical c=p: 0 (no critical point; support touches zero, Theorem 12(iii)).
    Edge = c·z(u*) with z'(u*)=0."""
    y = 1.0 / c
    # (0,∞) branch
    us = np.concatenate([np.linspace(1e-4, 0.05, 200), np.linspace(0.05, 3.0, 400), np.linspace(3.0, 50.0, 300)])
    vals = np.array([zp_of(u, y) for u in us])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(idx) > 0:
        j = idx[0]
        u_star = brentq(zp_of, us[j], us[j + 1], args=(y,), xtol=1e-14)
        return c * z_of(u_star, y)
    # (−∞,−1) branch
    usn = -np.logspace(6, 0, 2000, base=10)
    usn = usn[usn < -1.0]
    valsn = np.array([zp_of(u, y) for u in usn])
    idxn = np.where(np.sign(valsn[:-1]) != np.sign(valsn[1:]))[0]
    if len(idxn) > 0:
        j = idxn[-1]
        u_star = brentq(zp_of, usn[j], usn[j + 1], args=(y,), xtol=1e-12)
        return c * z_of(u_star, y)
    return 0.0

note("=== Prop 13 true lower edge (Poi(1) count law) ===")
edge_true = {c: true_lower_edge(c) for c in Cs}
for c in Cs:
    bench = (np.sqrt(c) - np.sqrt(p)) ** 2
    note(f"c={c:5.3f}  benchmark {bench:.5f}   true edge {edge_true[c]:.5f}   (true ≥ bench: {edge_true[c] >= bench - 1e-9})")

# ---------------- Part B: finite-sample λ_min⁺ ----------------
note("=== Part B: λ_min⁺, N=1000, 15 reps per c ===")
N_B, R_B = 1000, 15
edge_rows = []
for c in Cs:
    dB = int(round(c * N_B))
    w = np.full(N_B, 1.0 / N_B)
    lmins = []
    for r in range(R_B):
        n = rng.multinomial(N_B, w)
        I = np.nonzero(n)[0]
        M = len(I)
        Z = rng.standard_normal((dB, M))
        Bm = (Z * np.sqrt(n[I])) / np.sqrt(N_B)
        if M <= dB:
            G = Bm.T @ Bm
            lam_pos = np.linalg.eigvalsh(G)[0]
        else:
            G = Bm @ Bm.T
            lam_pos = np.linalg.eigvalsh(G)[0]
        lmins.append(lam_pos)
    lmins = np.array(lmins)
    bench = (np.sqrt(c) - np.sqrt(p)) ** 2
    edge_rows.append(dict(c=c, lmin_med=np.median(lmins), lmin_q1=np.percentile(lmins, 25),
                          lmin_q3=np.percentile(lmins, 75), bench=bench, edge_true=edge_true[c]))
    note(f"c={c:5.3f}  λ_min⁺ med {np.median(lmins):.5f} [q1 {np.percentile(lmins,25):.5f}, q3 {np.percentile(lmins,75):.5f}]  bench {bench:.5f}  true {edge_true[c]:.5f}")

# ---------------- pass checks ----------------
for row in edge_rows:
    c = row["c"]
    # median not below universal bound (small finite-size slack near criticality)
    slack = 0.02 if abs(c - p) < 0.1 else 0.005
    assert row["lmin_med"] >= row["bench"] - slack, (c, row["lmin_med"], row["bench"])
    assert row["edge_true"] >= row["bench"] - 1e-9
# Part A near theory (critical band smoothing allowed)
for row in atom_rows:
    tol = max(3 * row["q0_se"], 0.02)
    assert abs(row["q0_emp"] - row["q0_th"]) <= tol, row

with open(os.path.join(OUT, "E2_atom.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(atom_rows[0].keys()))
    wr.writeheader(); wr.writerows(atom_rows)
with open(os.path.join(OUT, "E2_edge.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(edge_rows[0].keys()))
    wr.writeheader(); wr.writerows(edge_rows)

# ---------------- figure ----------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
cc = [r["c"] for r in atom_rows]
axL.plot(cc, [r["q0_th"] for r in atom_rows], "r-", lw=1.5, label="theory max(0, 1−(1−e⁻¹)/c)")
axL.errorbar(cc, [r["q0_emp"] for r in atom_rows], yerr=[2 * r["q0_se"] for r in atom_rows],
             fmt="ko", ms=4, capsize=3, label="empirical (Part A, count rank)")
axL.axvline(p, color="gray", ls=":", lw=1); axL.text(p + 0.005, 0.3, "c*=1−e⁻¹", fontsize=8)
axL.set_xlabel("c"); axL.set_ylabel("zero mass"); axL.legend(fontsize=8)
axL.set_title("E2 Part A: rank/zero-atom transition (uniform, multinomial)")
axR.plot(cc, [r["bench"] for r in edge_rows], "g--", lw=1.5, label="universal bound |√c−√p|²")
axR.plot(cc, [r["edge_true"] for r in edge_rows], "r-", lw=1.5, label="true lower edge (Prop 13)")
axR.errorbar(cc, [r["lmin_med"] for r in edge_rows],
             yerr=[[r["lmin_med"] - r["lmin_q1"] for r in edge_rows],
                   [r["lmin_q3"] - r["lmin_med"] for r in edge_rows]],
             fmt="ko", ms=4, capsize=3, label="λ_min⁺ median & quartiles (Part B)")
axR.axvline(p, color="gray", ls=":", lw=1)
axR.set_xlabel("c"); axR.set_ylabel("lower edge / λ_min⁺"); axR.legend(fontsize=8)
axR.set_title("E2 Part B: lower edge — three levels")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "E2_figure.png"), dpi=150); plt.close(fig)

with open(os.path.join(OUT, "E2_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
note("=== E2 PASSED ===")
