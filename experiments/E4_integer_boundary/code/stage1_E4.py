# stage1_E4.py
# E4 — Residual integer-boundary sensitivity (Theorem 5 classification), count-only + one spectral panel.
# Per 实验指导手册 §四 E4 & §六 E4.
# Three constructions with ρ_N ⇒ δ_1:
#   (i)  exact uniform: R_N/N = 0, ν_res = δ_1
#   (ii) interior-α family: fraction α_N→α at 1−ε_N, rest at 1+(α_N/(1−α_N))ε_N
#        exact identities: (1/N)Σλ² = 1 + (α_N/(1−α_N))ε_N²,  R_N/N = α_N
#   (iii) endpoint α=1: λ_i = 1−1/N (i<N), λ_N = 2−1/N, N=2^m; R_N/N → 1, ν_res → Poi(1)
# Display: (1/N)Σ(λ_i−1)² → 0  alongside  R_N/N → {0, α, 1}.
# Integer protocol: k stored as int64; R = N − Σk in integer arithmetic; λ never
# reconstructed from float N w_i (constructions set λ directly).
# Outputs: output/E4_counts.csv, E4_figure.png, E4_spectral.png, E4_log.txt

import numpy as np, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260915
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

ALPHA = 0.4
REPS = 200
Ns = [2 ** m for m in range(8, 15)]  # 256 … 16384

def build(construction, N):
    """Return λ (float64), k (int64), u (float64), R (int). λ set directly (never via float N w)."""
    if construction == "uniform":
        lam = np.ones(N)
    elif construction == "interior":
        aN = int(round(ALPHA * N))
        aN = min(max(aN, 1), N - 1)
        eps = N ** -0.5
        a = aN / N
        lam = np.concatenate([
            np.full(aN, 1.0 - eps),
            np.full(N - aN, 1.0 + (a / (1.0 - a)) * eps),
        ])
    elif construction == "endpoint":
        lam = np.concatenate([np.full(N - 1, 1.0 - 1.0 / N), np.array([2.0 - 1.0 / N])])
    k = np.floor(lam).astype(np.int64)
    u = lam - k
    R = int(N - k.sum())
    return lam, k, u, R

def theory(construction):
    """(R_N/N limit, count zero mass, count 2nd moment, E_πU) in the limit."""
    if construction == "uniform":
        return 0.0, 0.0, 1.0, 0.0
    if construction == "interior":
        return ALPHA, ALPHA * np.exp(-1.0), 1.0 + ALPHA, ALPHA
    if construction == "endpoint":
        return 1.0, np.exp(-1.0), 2.0, 1.0

def run(construction, N, reps):
    lam, k, u, R = build(construction, N)
    lam2_dev = float(np.mean((lam - 1.0) ** 2))
    lam2_mean = float(np.mean(lam ** 2))
    summary = (float(lam.min()), float(lam.max()), float(np.max(np.abs(lam - lam.mean()))))
    if R > 0:
        probs = u / R
        zfrac = np.empty(reps)
        m2 = np.empty(reps)
        for r in range(reps):
            n = k + rng.multinomial(R, probs)
            zfrac[r] = np.mean(n == 0)
            m2[r] = np.mean(n.astype(float) ** 2)
        return R / N, lam2_dev, lam2_mean, zfrac.mean(), zfrac.std(ddof=1) / np.sqrt(reps), m2.mean(), m2.std(ddof=1) / np.sqrt(reps), summary
    else:
        # deterministic counts n_i = k_i: zero randomization
        n = k
        return R / N, lam2_dev, lam2_mean, float(np.mean(n == 0)), 0.0, float(np.mean(n.astype(float) ** 2)), 0.0, summary

log = []
def note(s):
    print(s); log.append(s)

rows = []
for cons in ["uniform", "interior", "endpoint"]:
    tR, tzero, tm2, tEU = theory(cons)
    note(f"--- {cons}: theory R/N={tR}, zero={tzero:.6f}, m2={tm2:.6f}, E U={tEU} ---")
    for N in Ns:
        RnN, dev2, lam2m, z, zse, m2, m2se, summ = run(cons, N, REPS)
        rows.append(dict(cons=cons, N=N, RnN=RnN, dev2=dev2, lam2_mean=lam2m,
                         zero=z, zero_se=zse, m2=m2, m2_se=m2se,
                         lam_min=summ[0], lam_max=summ[1], lam_maxdev=summ[2],
                         th_RnN=tR, th_zero=tzero, th_m2=tm2))
        note(f"N={N:6d}  R/N={RnN:.4f} (th {tR})  (1/N)Σ(λ−1)²={dev2:.3e}  zero={z:.4f}±{zse:.4f} (th {tzero:.4f})  m2={m2:.4f}±{m2se:.4f} (th {tm2:.4f})")

with open(os.path.join(OUT, "E4_counts.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)

# ---- pass checks ----------------------------------------------------------
for cons in ["uniform", "interior", "endpoint"]:
    tR, tzero, tm2, _ = theory(cons)
    sub = [r for r in rows if r["cons"] == cons]
    # R/N converges to target with shrinking band
    assert abs(sub[-1]["RnN"] - tR) <= abs(sub[0]["RnN"] - tR) + 1e-12
    # zero mass & m2 vs theory at largest N (within 5 SE or 0.02 abs)
    big = sub[-1]
    assert abs(big["zero"] - tzero) < max(5 * big["zero_se"], 0.02), (cons, big["zero"], tzero)
    assert abs(big["m2"] - tm2) < max(5 * big["m2_se"], 0.02), (cons, big["m2"], tm2)
    # second-order weight deviation vanishes (or is identically zero)
    assert sub[-1]["dev2"] <= sub[0]["dev2"] + 1e-18
note("checks: R/N limits, zero mass, m2, dev2 monotone — OK")

# ---- juxtaposition figure -------------------------------------------------
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax2 = ax1.twinx()
markers = {"uniform": "o", "interior": "s", "endpoint": "^"}
for cons in ["uniform", "interior", "endpoint"]:
    sub = [r for r in rows if r["cons"] == cons]
    Nl = [r["N"] for r in sub]
    ax1.semilogx(Nl, [max(r["dev2"], 1e-18) for r in sub], markers[cons] + "--", label=f"(1/N)Σ(λ−1)² [{cons}]", alpha=0.8)
    ax2.semilogx(Nl, [r["RnN"] for r in sub], markers[cons] + "-", label=f"R_N/N [{cons}]", alpha=0.9)
for y, name in [(0.0, "0"), (ALPHA, "α=0.4"), (1.0, "1")]:
    ax2.axhline(y, color="gray", lw=0.5, ls=":")
ax1.set_xlabel("N (log scale)"); ax1.set_ylabel("(1/N) Σ (λ_i − 1)²  (log)")
ax2.set_ylabel("R_N / N"); ax2.set_ylim(-0.05, 1.1)
ax1.set_yscale("log")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
ax1.set_title("E4: second-order weight deviation → 0, but residual randomization R_N/N → {0, α, 1}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "E4_figure.png"), dpi=150); plt.close(fig)

# ---- spectral panel (medium size): interior α=0.5, N=1024, c≈0.8 ----------
import stage0_solvers as S0
A2 = 0.5
N = 1024; d = 819; c = d / N
aN = int(round(A2 * N)); a = aN / N
eps = N ** -0.5
lam = np.concatenate([np.full(aN, 1.0 - eps), np.full(N - aN, 1.0 + (a / (1 - a)) * eps)])
k = np.floor(lam).astype(np.int64); u = lam - k; R = int(N - k.sum())
w = lam / N
X = rng.standard_normal((d, N))
spectra = {}
for scheme in ["pre", "multinomial", "residual"]:
    if scheme == "pre":
        n = lam
    elif scheme == "multinomial":
        n = rng.multinomial(N, w).astype(float)
    else:
        n = (k + rng.multinomial(R, u / R)).astype(float)
    S = (X * n) @ X.T / N
    spectra[scheme] = np.linalg.eigvalsh(S)

# theory curves
K = 60
ks_poi, p_poi, _, _ = S0.mixpoi_law([1.0], [1.0], K)                 # Poi(1)
ks_res = np.arange(K + 1)
p_res = 0.5 * (ks_res == 1) + 0.5 * p_poi                            # 0.5 δ1 + 0.5 Poi(1)
xgrid = np.linspace(0.02, 4.0, 500)
f_pre, _ = S0.density_grid(xgrid, 1e-3, c, ks_poi.astype(float) * 0 + 1.0, np.array([1.0]))
f_mult, _ = S0.density_grid(xgrid, 1e-3, c, ks_poi.astype(float), p_poi)
f_res, _ = S0.density_grid(xgrid, 1e-3, c, ks_res.astype(float), p_res)
nu0_mult = p_poi[0]; nu0_res = p_res[0]
mu0_mult = max(0, 1 - (1 - nu0_mult) / c); mu0_res = max(0, 1 - (1 - nu0_res) / c)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
for ax, scheme, fth, mu0, ttl in [
    (axes[0], "pre", f_pre, max(0, 1 - 1 / c), f"pre (MP(c), atom={max(0,1-1/c):.3f})"),
    (axes[1], "multinomial", f_mult, mu0_mult, f"multinomial (atom={mu0_mult:.3f})"),
    (axes[2], "residual", f_res, mu0_res, f"residual α=0.5 (atom={mu0_res:.3f})"),
]:
    pos = spectra[scheme][spectra[scheme] > 1e-10]
    ax.hist(pos, bins=100, density=True, alpha=0.6, label="empirical (positive part)")
    ax.plot(xgrid, fth, "r-", lw=1.5, label="T_c prediction")
    nzero = np.mean(spectra[scheme] <= 1e-10)
    ax.plot([0], [nzero], "ks", ms=8, label=f"zero mass emp={nzero:.3f}")
    ax.plot([0], [mu0], "r^", ms=8, label=f"zero mass th={mu0:.3f}")
    ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7)
fig.suptitle(f"E4 spectral panel: N={N}, d={d}, c={c:.4f}, interior α=0.5 construction")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "E4_spectral.png"), dpi=150); plt.close(fig)

with open(os.path.join(OUT, "E4_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
note("=== E4 PASSED ===")
