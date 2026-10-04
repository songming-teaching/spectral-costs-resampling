# stage1_T1.py
# T1 — Finite-sample moment-identity verification (Proposition 2) + Corollary 19 ordering.
# Per 实验指导手册 §四 T1 & §六 T1.
# Fixed (X, w): d=150, N=300, R=1000 resampling-only repetitions.
# Theory (multinomial): cost = (1/N)[Σ w_i‖x_i‖⁴ − tr S_{w,N}²]
# Theory (residual)   : cost = (1/N²) tr[(G∘G)(diag(u) − u uᵀ/R_N)], R_N=0 branch defined as 0.
# Shortcut: tr S_r² = (1/N²) nᵀ (G∘G) n — no eigendecomposition.
# Outputs: output/T1_table.csv, output/T1_log.txt

import numpy as np, os, csv

SEED = 20260915
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

d, N, R = 150, 300, 1000
rng = np.random.default_rng(SEED)

# ---- fixed (X, w) ------------------------------------------------------
X = rng.standard_normal((d, N))
w = rng.dirichlet(np.ones(N))           # non-uniform weights; guarantees R_N > 0 w.p. 1
lam = N * w
k = np.floor(lam).astype(np.int64)
u = lam - k
R_N = int(N - k.sum())
assert R_N > 0
assert abs(k.sum() + u.sum() - N) < 1e-9

G = X.T @ X
A = G * G                                 # G ∘ G
diagA = np.diag(A).copy()
trSw2 = float(lam @ A @ lam) / N ** 2     # tr S_{w,N}²

# ---- theoretical costs ---------------------------------------------------
theory_multi = (np.sum(w * diagA) - trSw2) / N
Cov_multi = N * (np.diag(w) - np.outer(w, w))
Cov_res = np.diag(u) - np.outer(u, u) / R_N
theory_res = (np.sum(u * diagA) - u @ A @ u / R_N) / N ** 2

# Corollary 19: Cov_multi − Cov_res ⪰ 0  ⇒  theory_multi ≥ theory_res ≥ 0
diff = Cov_multi - Cov_res
emin = float(np.linalg.eigvalsh(diff)[0])

# ---- Monte Carlo ---------------------------------------------------------
cost_multi = np.empty(R)
cost_res = np.empty(R)
for r in range(R):
    n_m = rng.multinomial(N, w)
    n_r = k + rng.multinomial(R_N, u / R_N)
    cost_multi[r] = n_m @ A @ n_m / N ** 2 - trSw2
    cost_res[r] = n_r @ A @ n_r / N ** 2 - trSw2

def report(cost, theory):
    mean = cost.mean()
    se = cost.std(ddof=1) / np.sqrt(len(cost))
    z = abs(mean - theory) / se
    return mean, se, z

rows = []
for name, cost, theory in [("multinomial", cost_multi, theory_multi),
                           ("residual", cost_res, theory_res)]:
    mean, se, z = report(cost, theory)
    rows.append((name, mean, se, theory, z))

log = []
def note(s):
    print(s); log.append(s)

note(f"T1: d={d}, N={N}, R={R}, R_N={R_N}")
note(f"tr S_w² = {trSw2:.6f}")
note(f"theory cost multi = {theory_multi:.6e}, theory cost res = {theory_res:.6e}")
note(f"Corollary 19: eigmin(Cov_multi − Cov_res) = {emin:.3e} (should be ≥ −1e-12)")
assert emin > -1e-9, "Corollary 19 PSD relation violated"
assert theory_multi >= theory_res >= 0
note("scheme        MC mean        MC SE          theory         |MC−theory|/SE")
ok = True
for name, mean, se, theory, z in rows:
    note(f"{name:12s}  {mean: .6e}   {se:.3e}   {theory: .6e}   {z:.3f}")
    ok &= (z < 3.0)
assert ok, "standardized deviation exceeds 3"

# ordering at MC level
note(f"MC ordering: mean_res ({cost_res.mean():.6e}) ≤ mean_multi ({cost_multi.mean():.6e}): {cost_res.mean() <= cost_multi.mean()}")

with open(os.path.join(OUT, "T1_table.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["scheme", "MC_mean", "MC_SE", "theory", "standardized_dev"])
    for name, mean, se, theory, z in rows:
        wr.writerow([name, f"{mean:.10e}", f"{se:.6e}", f"{theory:.10e}", f"{z:.4f}"])
    wr.writerow([])
    wr.writerow(["trSw2", trSw2, "R_N", R_N, "eigmin_diff", emin])
with open(os.path.join(OUT, "T1_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
note("=== T1 PASSED ===")
