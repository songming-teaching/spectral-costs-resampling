# stage2_E3.py
# E3 — Same ESS, different asymptotic nullities (Proposition 10).
# ρA = ½δ_{1/4}+½δ_{7/4}, ρB = (9/10)δ_{3/4}+(1/10)δ_{13/4}; ESS exactly 16N/25 for both.
# N=1000 (multiple of 10), c=0.8, d=800, R=30, multinomial resampling.
# Preregistered: m2 identical (3.05); count zero masses 0.476287 vs 0.429007;
# spectral zero atoms 0.345359 vs 0.286259 (gap ≥ 4 points, direction A > B).
# Outputs: output/E3_four_quantities.csv, E3_figure.png, E3_log.txt

import numpy as np, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stage0_solvers as S0

SEED = 20260915
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

c = 0.8
N, d, R = 1000, 800, 30
K = 60

PROFILES = {
    "A": dict(lam=np.concatenate([np.full(500, 0.25), np.full(500, 1.75)]),
              pts=[0.25, 1.75], ws=[0.5, 0.5],
              preg=dict(nu0=0.476287, mu0=0.345359, m2=3.05, b3=8.375)),
    "B": dict(lam=np.concatenate([np.full(900, 0.75), np.full(100, 3.25)]),
              pts=[0.75, 3.25], ws=[0.9, 0.1],
              preg=dict(nu0=0.429007, mu0=0.286259, m2=3.05, b3=9.5)),
}

log = []
def note(s):
    print(s); log.append(s)

# ---- ESS assertion (exact equality) ----
for name, prof in PROFILES.items():
    lam = prof["lam"]
    ess = N ** 2 / float(np.sum(lam ** 2))
    prof["ess"] = ess
    note(f"profile {name}: ESS = {ess:.6f}")
assert abs(PROFILES["A"]["ess"] - PROFILES["B"]["ess"]) < 1e-9
assert abs(PROFILES["A"]["ess"] - 16 * N / 25) < 1e-9

rows = []
emp = {}
for name, prof in PROFILES.items():
    lam = prof["lam"]
    w = lam / N
    ks, pb, _, _ = S0.mixpoi_law(prof["pts"], prof["ws"], K)
    prof["law"] = (ks.astype(float), pb)
    m2s, nu0s, mu0s, m3s = [], [], [], []
    eigs_all = []
    for r in range(R):
        X = rng.standard_normal((d, N))
        n = rng.multinomial(N, w).astype(float)
        S = (X * n) @ X.T / N
        eigs = np.linalg.eigvalsh(S)
        eigs_all.append(eigs)
        M = int(np.count_nonzero(n))
        m2s.append(float((eigs ** 2).mean()))
        m3s.append(float((eigs ** 3).mean()))
        nu0s.append(float(np.mean(n == 0)))
        mu0s.append((d - min(d, M)) / d)
    emp[name] = dict(m2=np.array(m2s), nu0=np.array(nu0s), mu0=np.array(mu0s), m3=np.array(m3s),
                     eigs=np.concatenate(eigs_all))
    pr = prof["preg"]
    m3_th = 1 + 3 * c * (1 + 25 / 16) + c ** 2 * pr["b3"]
    rows.append(dict(profile=name, ess=prof["ess"],
                     m2_emp=emp[name]["m2"].mean(), m2_se=emp[name]["m2"].std(ddof=1) / np.sqrt(R), m2_th=pr["m2"],
                     nu0_emp=emp[name]["nu0"].mean(), nu0_se=emp[name]["nu0"].std(ddof=1) / np.sqrt(R), nu0_th=pr["nu0"],
                     mu0_emp=emp[name]["mu0"].mean(), mu0_se=emp[name]["mu0"].std(ddof=1) / np.sqrt(R), mu0_th=pr["mu0"],
                     m3_emp=emp[name]["m3"].mean(), m3_se=emp[name]["m3"].std(ddof=1) / np.sqrt(R), m3_th=m3_th))
    note(f"profile {name}: m2 {rows[-1]['m2_emp']:.4f}±{rows[-1]['m2_se']:.4f} (th {pr['m2']}) | "
         f"nu0 {rows[-1]['nu0_emp']:.5f}±{rows[-1]['nu0_se']:.5f} (th {pr['nu0']}) | "
         f"mu0 {rows[-1]['mu0_emp']:.5f}±{rows[-1]['mu0_se']:.5f} (th {pr['mu0']}) | "
         f"m3 {rows[-1]['m3_emp']:.3f}±{rows[-1]['m3_se']:.3f} (th {m3_th:.3f})")

# ---- pass checks ----
rA, rB = rows
assert abs(rA["m2_emp"] - rA["m2_th"]) <= max(2 * rA["m2_se"], 0.01)
assert abs(rB["m2_emp"] - rB["m2_th"]) <= max(2 * rB["m2_se"], 0.01)
# m2 empirically indistinguishable (within joint 2σ)
se_joint = np.sqrt(rA["m2_se"] ** 2 + rB["m2_se"] ** 2)
note(f"|m2_A − m2_B| = {abs(rA['m2_emp']-rB['m2_emp']):.4f}, joint 2σ = {2*se_joint:.4f}")
assert abs(rA["m2_emp"] - rB["m2_emp"]) <= 2 * se_joint
# zero-atom gap ≥ 4 points, correct direction
gap = rA["mu0_emp"] - rB["mu0_emp"]
note(f"spectral zero-atom gap A−B = {gap:.4f} (th {rA['mu0_th']-rB['mu0_th']:.4f}; ≥ 0.04 required)")
assert gap >= 0.04
for r in rows:
    assert abs(r["nu0_emp"] - r["nu0_th"]) <= max(2 * r["nu0_se"], 0.01)
    assert abs(r["mu0_emp"] - r["mu0_th"]) <= max(2 * r["mu0_se"], 0.01)

with open(os.path.join(OUT, "E3_four_quantities.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)

# ---- figure: CDF comparison + zero-mass bars ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
for name, color in [("A", "C0"), ("B", "C3")]:
    e = np.sort(emp[name]["eigs"])
    cdf = np.arange(1, len(e) + 1) / len(e)
    axL.plot(e, cdf, color=color, lw=1.2, label=f"empirical CDF [{name}]")
    axL.axvline(0, color="gray", lw=0.5)
axL.set_xlim(-0.2, 6); axL.set_xlabel("eigenvalue"); axL.set_ylabel("CDF")
axL.set_title(f"E3: same ESS = {PROFILES['A']['ess']:.0f}, different spectra")
axL.legend(fontsize=8)
names = ["A", "B"]
emp_mu = [rows[0]["mu0_emp"], rows[1]["mu0_emp"]]
th_mu = [rows[0]["mu0_th"], rows[1]["mu0_th"]]
xpos = np.arange(2)
axR.bar(xpos - 0.15, emp_mu, width=0.3, label="empirical μ({0})")
axR.bar(xpos + 0.15, th_mu, width=0.3, label="theory μ({0})")
for i, (e_, t_) in enumerate(zip(emp_mu, th_mu)):
    axR.text(i - 0.15, e_ + 0.005, f"{e_:.3f}", ha="center", fontsize=9)
    axR.text(i + 0.15, t_ + 0.005, f"{t_:.3f}", ha="center", fontsize=9)
axR.set_xticks(xpos); axR.set_xticklabels([f"profile {n}\n(count ν0: {rows[i]['nu0_th']:.4f})" for i, n in enumerate(names)])
axR.set_ylabel("spectral zero atom"); axR.legend(fontsize=8)
axR.set_title(f"zero-atom gap = {gap:.3f} (th {rA['mu0_th']-rB['mu0_th']:.3f})")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "E3_figure.png"), dpi=150); plt.close(fig)

with open(os.path.join(OUT, "E3_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
note("=== E3 PASSED ===")
