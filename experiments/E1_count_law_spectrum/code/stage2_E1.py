# stage2_E1.py
# E1 — Count-law and spectral-transform verification (Theorems 3,4,6,14; Corollary 18).
# Per 实验指导手册 §四 E1 & §六 E1. c=0.8, H=δ1, N=1000, d=800, R=20, paired clouds.
# Profiles: exact uniform; two-point ½δ_{1/2}+½δ_{3/2} (exact ratio construction).
# Outputs: output/E1_moments.csv, E1_figure_uniform.png, E1_figure_twopoint.png,
#          E1_stieltjes.csv, E1_log.txt

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
N, d, R = 1000, 800, 20
K = 60

log = []
def note(s):
    print(s); log.append(s)

# ---------------- theoretical count laws ----------------
ksU, pU_mult, _, _ = S0.mixpoi_law([1.0], [1.0], K)                    # Poi(1)
pU_res = (np.arange(K + 1) == 1).astype(float)                         # δ1
ks2, p2_mult, _, _ = S0.mixpoi_law([0.5, 1.5], [0.5, 0.5], K)
ks2r, p2_res, _, _ = S0.residual_twopoint_half(K)                      # ½Poi(.5)+½(1+Poi(.5))

PROFILES = {
    "uniform": dict(
        lam=np.ones(N),
        laws={"multinomial": (ksU.astype(float), pU_mult), "residual": (ksU.astype(float), pU_res)},
        preg=dict(r2=1.0, ubar=0.0, m2_pre=1.8, m2_mult=2.6, m2_res=1.8,
                  nu0_mult=0.367879, nu0_res=0.0, mu0_mult=0.20985, mu0_res=0.0),
    ),
    "twopoint": dict(
        lam=np.concatenate([np.full(N // 2, 0.5), np.full(N - N // 2, 1.5)]),
        laws={"multinomial": (ks2.astype(float), p2_mult), "residual": (ks2r.astype(float), p2_res)},
        preg=dict(r2=1.25, ubar=0.5, m2_pre=2.0, m2_mult=2.8, m2_res=2.4,
                  nu0_mult=0.41483, nu0_res=0.30327, mu0_mult=0.26854, mu0_res=0.12909),
    ),
}

# ---------------- helpers ----------------
def spectra_and_counts(lam, rng):
    """One repetition: shared cloud X; returns dict of spectra and count vectors."""
    X = rng.standard_normal((d, N))
    w = lam / N
    k = np.floor(lam).astype(np.int64)
    u = lam - k
    R_N = int(N - k.sum())
    out = {}
    for scheme in ["pre", "multinomial", "residual"]:
        if scheme == "pre":
            n = lam.astype(float)
        elif scheme == "multinomial":
            n = rng.multinomial(N, w).astype(float)
        else:
            n = (k + (rng.multinomial(R_N, u / R_N) if R_N > 0 else 0)).astype(float)
        S = (X * n) @ X.T / N
        eigs = np.linalg.eigvalsh(S)
        M = int(np.count_nonzero(n))
        mu0 = (d - min(d, M)) / d
        out[scheme] = dict(eigs=eigs, n=n, m1=float(eigs.mean()), m2=float((eigs ** 2).mean()),
                           nu0=float(np.mean(n == 0)), mu0=mu0)
    return out

def theory_m(c_law, supp, probs):
    return S0.solve_stieltjes_cov

mom_rows = []
sth_rows = []
all_curves = {}

xgrid = np.linspace(0.05, 6.0, 300)
h_plot = 1e-3
zgrid_Eh = np.linspace(0.05, 8.0, 40) + 1j * 0.05

for pname, prof in PROFILES.items():
    lam = prof["lam"]
    preg = prof["preg"]
    note(f"===== profile {pname} (N={N}, d={d}, c={c}, R={R}) =====")
    reps = [spectra_and_counts(lam, rng) for _ in range(R)]
    # theoretical curves
    curves = {}
    curves["pre"] = S0.density_grid(xgrid, h_plot, c, lam[:1] * 0 + (1.0 if pname == "uniform" else None), None) if False else None
    if pname == "uniform":
        supp_pre, p_pre = np.array([1.0]), np.array([1.0])
    else:
        supp_pre, p_pre = np.array([0.5, 1.5]), np.array([0.5, 0.5])
    curves = {"pre": S0.density_grid(xgrid, h_plot, c, supp_pre, p_pre)[0]}
    for scheme in ["multinomial", "residual"]:
        sp, pb = prof["laws"][scheme]
        curves[scheme] = S0.density_grid(xgrid, h_plot, c, sp, pb)[0]
    all_curves[pname] = curves
    # moments table
    for scheme in ["pre", "multinomial", "residual"]:
        m1s = np.array([r[scheme]["m1"] for r in reps])
        m2s = np.array([r[scheme]["m2"] for r in reps])
        nu0s = np.array([r[scheme]["nu0"] for r in reps])
        mu0s = np.array([r[scheme]["mu0"] for r in reps])
        if scheme == "pre":
            m2_th, nu0_th, mu0_th = preg["m2_pre"], 0.0, 0.0
        else:
            m2_th = preg[f"m2_{'mult' if scheme=='multinomial' else 'res'}"]
            nu0_th = preg[f"nu0_{'mult' if scheme=='multinomial' else 'res'}"]
            mu0_th = preg[f"mu0_{'mult' if scheme=='multinomial' else 'res'}"]
        row = dict(profile=pname, scheme=scheme,
                   m1_emp=m1s.mean(), m1_se=m1s.std(ddof=1) / np.sqrt(R), m1_th=1.0,
                   m2_emp=m2s.mean(), m2_se=m2s.std(ddof=1) / np.sqrt(R), m2_th=m2_th,
                   nu0_emp=nu0s.mean(), nu0_se=nu0s.std(ddof=1) / np.sqrt(R), nu0_th=nu0_th,
                   mu0_emp=mu0s.mean(), mu0_se=mu0s.std(ddof=1) / np.sqrt(R), mu0_th=mu0_th)
        mom_rows.append(row)
        note(f"{scheme:12s} m1 {row['m1_emp']:.4f}±{row['m1_se']:.4f} | m2 {row['m2_emp']:.3f}±{row['m2_se']:.3f} (th {m2_th}) | "
             f"nu0 {row['nu0_emp']:.4f}±{row['nu0_se']:.4f} (th {nu0_th}) | mu0 {row['mu0_emp']:.4f}±{row['mu0_se']:.4f} (th {mu0_th})")
    # pass checks: within ±2σ or ±1%
    for row in [r for r in mom_rows if r["profile"] == pname]:
        for key in ["m2", "nu0", "mu0"]:
            emp, se, th = row[f"{key}_emp"], row[f"{key}_se"], row[f"{key}_th"]
            tol = max(2 * se, 0.01 * max(abs(th), 0.01))
            assert abs(emp - th) <= tol, (pname, row["scheme"], key, emp, th, tol)
    # Stieltjes discrepancy E_h (pooled eigenvalues over reps)
    for scheme in ["pre", "multinomial", "residual"]:
        pooled = np.concatenate([r[scheme]["eigs"] for r in reps])
        m_emp = np.mean(1.0 / (pooled[None, :] - zgrid_Eh[:, None]), axis=1)
        sp, pb = (supp_pre, p_pre) if scheme == "pre" else prof["laws"][scheme]
        m_th = S0._solve_grid(zgrid_Eh, c, sp, pb, comp=False)
        Eh_max = float(np.max(np.abs(m_emp - m_th)))
        Eh_mean = float(np.mean(np.abs(m_emp - m_th)))
        z2 = np.linspace(0.05, 8.0, 40) + 1j * 0.025
        m_emp2 = np.mean(1.0 / (pooled[None, :] - z2[:, None]), axis=1)
        m_th2 = S0._solve_grid(z2, c, sp, pb, comp=False)
        Eh2_max = float(np.max(np.abs(m_emp2 - m_th2)))
        sth_rows.append(dict(profile=pname, scheme=scheme, h=0.05, Eh_max=Eh_max, Eh_mean=Eh_mean, Eh_h2_max=Eh2_max))
        note(f"  E_h({scheme}): max={Eh_max:.4f}, mean={Eh_mean:.4f}, h/2 max={Eh2_max:.4f}")

# ---------------- write moments csv ----------------
with open(os.path.join(OUT, "E1_moments.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(mom_rows[0].keys()))
    wr.writeheader(); wr.writerows(mom_rows)
with open(os.path.join(OUT, "E1_stieltjes.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(sth_rows[0].keys()))
    wr.writeheader(); wr.writerows(sth_rows)

# ---------------- figures ----------------
for pname, prof in PROFILES.items():
    preg = prof["preg"]
    curves = all_curves[pname]
    reps_plot = [spectra_and_counts(prof["lam"], rng) for _ in range(R)]  # fresh pooled for plots
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    # row 0: count PMFs
    for j, scheme in enumerate(["multinomial", "residual"]):
        ax = axes[0, j if scheme == "multinomial" else 2]
        pooled_n = np.concatenate([r[scheme]["n"] for r in reps_plot])
        vals, cnts = np.unique(pooled_n, return_counts=True)
        ax.stem(vals, cnts / len(pooled_n), linefmt="C0-", markerfmt="C0o", basefmt=" ", label="empirical")
        sp, pb = prof["laws"][scheme]
        ax.stem(sp, pb, linefmt="r--", markerfmt="rx", basefmt=" ", label="theory")
        ax.set_title(f"count PMF [{pname}] {scheme}")
        ax.legend(fontsize=8); ax.set_xlim(-0.5, 8.5)
    axes[0, 1].axis("off")
    # row 1: spectra
    for j, scheme in enumerate(["pre", "multinomial", "residual"]):
        ax = axes[1, j]
        pooled = np.concatenate([r[scheme]["eigs"] for r in reps_plot])
        pos = pooled[pooled > 1e-10]
        ax.hist(pos, bins=120, density=True, alpha=0.55, label="empirical")
        ax.plot(xgrid, curves[scheme], "r-", lw=1.3, label="T_c prediction")
        mu0_emp = np.mean(pooled <= 1e-10)
        mu0_th = 0.0 if scheme == "pre" else preg[f"mu0_{'mult' if scheme=='multinomial' else 'res'}"]
        ax.plot([0], [mu0_emp], "ks", ms=8, label=f"atom emp={mu0_emp:.3f}")
        ax.plot([0], [mu0_th], "r^", ms=8, label=f"atom th={mu0_th:.3f}")
        ax.set_title(f"spectrum [{pname}] {scheme}")
        ax.legend(fontsize=8); ax.set_xlim(-0.3, 6)
    fig.suptitle(f"E1 [{pname}]: count laws and T_c spectral transforms (N={N}, d={d}, c={c}, R={R})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"E1_figure_{pname}.png"), dpi=150)
    plt.close(fig)

with open(os.path.join(OUT, "E1_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
note("=== E1 PASSED ===")
