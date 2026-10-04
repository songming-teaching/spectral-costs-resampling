# stage3_appendix.py
# Appendix items: (A1) scale checks, (A2) non-Gaussian control,
# (A3) two-point population spectrum H ⊠ μ_Q (Theorem 7), (A4) solver sensitivity archive.
# Outputs: output/A1_scale.csv, A2_nongaussian.csv, A3_population.csv,
#          A3_population.png, A4_solver_archive.md, A_log.txt
#
# A3 theory curve via free multiplicative convolution (S-transform pipeline),
# validated on a control case (H ⊠ MP(c) vs Wishart-with-H simulation) before use.

import numpy as np, os, csv, json
from scipy.optimize import fsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stage0_solvers as S0

SEED = 20260915
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

c = 0.8
K = 60

# ==================== S-transform pipeline ====================
def psi_of(mfun, x):
    x = np.asarray(x, dtype=complex)
    return -(1.0 / x) * mfun(1.0 / x) - 1.0

def chi_of(mfun, w, seed=None):
    """Solve psi(x) = w for x (2D Newton via fsolve; multiple seeds)."""
    w = complex(w)
    def F(v):
        val = psi_of(mfun, v[0] + 1j * v[1])
        return [val.real - w.real, val.imag - w.imag]
    seeds = ([seed[0], seed[1]] if seed is not None else []) + \
            [[w.real, w.imag], [0.05, 0.05], [-0.05, 0.1], [1.0, 0.1], [-0.3, -0.3]]
    for sd in seeds:
        sol, info, ier, msg = fsolve(F, sd, full_output=True, xtol=1e-12)
        if ier == 1:
            x = sol[0] + 1j * sol[1]
            if abs(psi_of(mfun, x) - w) < 1e-8:
                return x
    raise RuntimeError(f"chi solve failed at w={w}")

def S_of(mfun, w, seed=None):
    w = complex(w)
    ch = chi_of(mfun, w, seed=seed)
    return (1.0 + w) / w * ch

def m_product(z0, mfun1, mfun2, seed=None):
    """Stieltjes of μ1 ⊠ μ2 at z0 via S-transform.
    Solve χ_prod(x) = 1/z0  ⟺  x·S_prod(x)/(1+x) = 1/z0; then m(z0) = −(x+1)/z0.
    seed: previous solution for warm continuation along a grid."""
    z0 = complex(z0)
    w = 1.0 / z0
    def F(v):
        y = v[0] + 1j * v[1]
        try:
            s = S_of(mfun1, y) * S_of(mfun2, y)
            val = y * s / (1.0 + y) - w
        except Exception:
            val = 1e6 + 0j
        return [val.real, val.imag]
    seeds = ([[seed[0], seed[1]]] if seed is not None else []) + \
            [[w.real, w.imag], [-0.3, -0.3], [0.01, 0.01], [-0.5, 0.5], [5.0, -0.5], [15.0, -1.0]]
    for sd in seeds:
        sol, info, ier, msg = fsolve(F, sd, full_output=True, xtol=1e-11)
        if ier == 1:
            y = sol[0] + 1j * sol[1]
            try:
                r = F([y.real, y.imag])
                if abs(r[0]) + abs(r[1]) < 1e-7:
                    return -(y + 1.0) / z0, y
            except Exception:
                pass
    raise RuntimeError(f"product solve failed at z={z0}")

# ---- mfun for MP(c) via closed-form quadratic ----
def mfun_MP(z):
    zz = np.atleast_1d(z)
    out = np.empty(len(zz), dtype=complex)
    for i, z in enumerate(zz):
        r = np.roots([c * z, z + c - 1, 1])
        sel = r[r.imag > 0]
        out[i] = sel[0] if len(sel) else r[np.argmax(r.imag)]
    return out if np.ndim(z) else complex(out[0])

def mfun_H(z):
    zz = np.atleast_1d(z)
    out = 0.5 / (0.5 - zz) + 0.5 / (1.5 - zz)
    return out if np.ndim(z) else complex(out[0])

def mfun_muQ(z):
    ks_poi, p_poi, _, _ = S0.mixpoi_law([1.0], [1.0], K)
    zz = np.atleast_1d(z)
    m = S0._solve_grid(zz, c, ks_poi.astype(float), p_poi, comp=False)
    return m if np.ndim(z) else complex(m[0])

def main():
    rng = np.random.default_rng(SEED)
    log = []
    def note(s):
        print(s); log.append(s)
    ks_poi, p_poi, _, _ = S0.mixpoi_law([1.0], [1.0], K)

    # ---- CONTROL: H boxtimes MP(c) vs simulation (Wishart with two-point population) ----
    note("=== A3 control: H boxtimes MP(c) pipeline validation ===")
    d_c, N_c = 2000, 2500
    sig_c = np.concatenate([np.full(d_c // 2, 0.5), np.full(d_c - d_c // 2, 1.5)])
    Z_c = rng.standard_normal((d_c, N_c))
    S_c = np.diag(np.sqrt(sig_c)) @ (Z_c @ Z_c.T / N_c) @ np.diag(np.sqrt(sig_c))
    eigs_c = np.linalg.eigvalsh(S_c)
    devs = []
    for z in [1j, 0.5 + 0.5j, 2 + 1j, 0.2 + 0.3j]:
        m_th, _ = m_product(z, mfun_H, mfun_MP)
        m_emp = np.mean(1.0 / (eigs_c - z))
        devs.append(abs(m_th - m_emp))
        note(f"  z={z}: pipeline m={m_th:.6f}, empirical={m_emp:.6f}, |diff|={abs(m_th-m_emp):.2e}")
    control_ok = max(devs) < 5e-3
    note(f"control max deviation = {max(devs):.2e} (<5e-3 required): {control_ok}")
    assert control_ok

    # ==================== A1: scale checks ====================
    note("=== A1: scale checks (N=500, 2000; uniform & two-point; R=10) ===")
    def one_scale(Nv, pname):
        d_v = int(round(0.8 * Nv))
        if pname == "uniform":
            lam = np.ones(Nv)
            th = dict(m2_mult=1 + c * 2, nu0=np.exp(-1), mu0=max(0, 1 - (1 - np.exp(-1)) / c))
        else:
            lam = np.concatenate([np.full(Nv // 2, 0.5), np.full(Nv - Nv // 2, 1.5)])
            nu0 = 0.5 * (np.exp(-0.5) + np.exp(-1.5))
            th = dict(m2_mult=1 + c * 2.25, nu0=nu0, mu0=max(0, 1 - (1 - nu0) / c))
        m2s, nu0s, mu0s = [], [], []
        for r in range(10):
            X = rng.standard_normal((d_v, Nv))
            n = rng.multinomial(Nv, lam / Nv).astype(float)
            S = (X * n) @ X.T / Nv
            eigs = np.linalg.eigvalsh(S)
            M = int(np.count_nonzero(n))
            m2s.append(float((eigs ** 2).mean()))
            nu0s.append(float(np.mean(n == 0)))
            mu0s.append((d_v - min(d_v, M)) / d_v)
        return dict(N=Nv, profile=pname, d=d_v,
                    m2=np.mean(m2s), m2_se=np.std(m2s, ddof=1) / 3.162, m2_th=th["m2_mult"],
                    nu0=np.mean(nu0s), nu0_se=np.std(nu0s, ddof=1) / 3.162, nu0_th=th["nu0"],
                    mu0=np.mean(mu0s), mu0_se=np.std(mu0s, ddof=1) / 3.162, mu0_th=th["mu0"])
    a1_rows = []
    for Nv in [500, 2000]:
        for pname in ["uniform", "twopoint"]:
            row = one_scale(Nv, pname)
            a1_rows.append(row)
            note(f"N={Nv} {pname}: m2 {row['m2']:.3f}±{row['m2_se']:.3f} (th {row['m2_th']}) | "
                 f"nu0 {row['nu0']:.4f}±{row['nu0_se']:.4f} (th {row['nu0_th']:.4f}) | "
                 f"mu0 {row['mu0']:.4f}±{row['mu0_se']:.4f} (th {row['mu0_th']:.4f})")
    with open(os.path.join(OUT, "A1_scale.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(a1_rows[0].keys()))
        wr.writeheader(); wr.writerows(a1_rows)

    # ==================== A2: non-Gaussian control ====================
    note("=== A2: non-Gaussian control (Rademacher & Student-t8; two-point, multinomial, N=1000, d=800, R=10) ===")
    N2, d2, R2 = 1000, 800, 10
    lam2 = np.concatenate([np.full(500, 0.5), np.full(500, 1.5)])
    a2_rows = []
    for dist in ["gaussian", "rademacher", "student-t8"]:
        m2s, mu0s = [], []
        for r in range(R2):
            if dist == "gaussian":
                X = rng.standard_normal((d2, N2)); kap4 = 0.0
            elif dist == "rademacher":
                X = rng.choice([-1.0, 1.0], size=(d2, N2)); kap4 = -2.0
            else:
                X = rng.standard_t(8, size=(d2, N2)) / np.sqrt(8 / (8 - 2)); kap4 = 6 / (8 - 4)
            n = rng.multinomial(N2, lam2 / N2).astype(float)
            S = (X * n) @ X.T / N2
            eigs = np.linalg.eigvalsh(S)
            M = int(np.count_nonzero(n))
            m2s.append(float((eigs ** 2).mean()))
            mu0s.append((d2 - min(d2, M)) / d2)
        m2_th_asym = 1 + c * 2.25
        m2_th_finite = 1 + c * 1.25 + (1 - np.sum((lam2 / N2) ** 2)) * (d2 + 1 + kap4) / N2
        a2_rows.append(dict(dist=dist, kappa4=kap4,
                            m2=np.mean(m2s), m2_se=np.std(m2s, ddof=1) / np.sqrt(R2),
                            m2_th_asymptotic=m2_th_asym, m2_th_finiteN=m2_th_finite,
                            mu0=np.mean(mu0s), mu0_se=np.std(mu0s, ddof=1) / np.sqrt(R2), mu0_th=0.26854))
        note(f"{dist:11s}: m2 {np.mean(m2s):.3f}±{np.std(m2s,ddof=1)/np.sqrt(R2):.3f} (asym {m2_th_asym}, finite-N {m2_th_finite:.3f}) | "
             f"mu0 {np.mean(mu0s):.4f}±{np.std(mu0s,ddof=1)/np.sqrt(R2):.4f} (th 0.26854)")
    note("A2 note: single configuration; agreement level reported as-is, NOT a universality proof.")
    with open(os.path.join(OUT, "A2_nongaussian.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(a2_rows[0].keys()))
        wr.writeheader(); wr.writerows(a2_rows)

    # ==================== A3: two-point population spectrum ====================
    note("=== A3: two-point population H = (1/2)delta_{1/2}+(1/2)delta_{3/2}, uniform weights, multinomial; mu_S = H boxtimes T_c(Poi(1)) ===")
    N3, d3, R3 = 1000, 800, 10
    sig3 = np.concatenate([np.full(d3 // 2, 0.5), np.full(d3 - d3 // 2, 1.5)])
    a1_H, m2H = 1.0, 1.25
    m2_pre_th = m2H + c * a1_H ** 2 * 1.0
    m2_post_th = m2H + c * a1_H ** 2 * 2.0
    m1s, m2s_pre, m2s_post, eigs_pool = [], [], [], []
    for r in range(R3):
        Z = rng.standard_normal((d3, N3))
        n = rng.multinomial(N3, np.full(N3, 1.0 / N3)).astype(float)
        Sg = np.diag(np.sqrt(sig3))
        e_post = np.linalg.eigvalsh(Sg @ ((Z * n) @ Z.T / N3) @ Sg)
        e_pre = np.linalg.eigvalsh(Sg @ ((Z * 1.0) @ Z.T / N3) @ Sg)
        eigs_pool.append(e_post)
        m1s.append(e_post.mean()); m2s_post.append((e_post ** 2).mean()); m2s_pre.append((e_pre ** 2).mean())
    note(f"m1 post {np.mean(m1s):.4f} (th 1.0) | m2 pre {np.mean(m2s_pre):.3f} (th {m2_pre_th}) | "
         f"m2 post {np.mean(m2s_post):.3f} (th {m2_post_th})")
    assert abs(np.mean(m2s_pre) - m2_pre_th) < 0.03
    assert abs(np.mean(m2s_post) - m2_post_th) < 0.03
    eigs_pool = np.concatenate(eigs_pool)
    mu0_emp = np.mean(eigs_pool <= 1e-10)
    note(f"zero atom: emp {mu0_emp:.4f} vs th 0.20985 (max-atom rule)")

    xgrid = np.linspace(0.02, 6.0, 250)
    f_prod = np.empty_like(xgrid)
    y_prev = None
    for j in range(len(xgrid) - 1, -1, -1):   # right-to-left warm continuation
        x = xgrid[j]
        mval, y_prev = m_product(x + 1j * 2e-3, mfun_H, mfun_muQ,
                                 seed=None if y_prev is None else [y_prev.real, y_prev.imag])
        f_prod[j] = mval.imag / np.pi
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos = eigs_pool[eigs_pool > 1e-10]
    ax.hist(pos, bins=120, density=True, alpha=0.55, label="empirical (positive part)")
    ax.plot(xgrid, f_prod, "r-", lw=1.4, label="H ⊠ μ_Q (S-transform pipeline)")
    ax.plot([0], [mu0_emp], "ks", ms=8, label=f"atom emp={mu0_emp:.3f}")
    ax.plot([0], [0.20985], "r^", ms=8, label="atom th=0.210")
    ax.set_title("A3: two-point population spectrum, H boxtimes T_c(Poi(1))")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "A3_population.png"), dpi=150); plt.close(fig)
    with open(os.path.join(OUT, "A3_population.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["m1_post_emp", float(np.mean(m1s)), "th", 1.0])
        wr.writerow(["m2_pre_emp", float(np.mean(m2s_pre)), "th", m2_pre_th])
        wr.writerow(["m2_post_emp", float(np.mean(m2s_post)), "th", m2_post_th])
        wr.writerow(["mu0_emp", float(mu0_emp), "th", 0.20985])
        wr.writerow(["control_max_deviation", max(devs)])

    # ==================== A4: solver sensitivity archive ====================
    with open(os.path.join(OUT, "stage0_checks.json"), encoding="utf-8") as f:
        s0 = json.load(f)
    with open(os.path.join(OUT, "A4_solver_archive.md"), "w", encoding="utf-8") as f:
        f.write("# A4 solver sensitivity archive\n\n")
        f.write("Consolidated from stage0 (see stage0_log.txt for full detail).\n\n")
        f.write(f"- MP(c) recovery, interior max relative error: {s0['MP_recovery_err_pct']:.3f}%\n")
        tr = s0["trunc"]
        f.write(f"- Truncation K=20: mass {tr['K20_mass']:.2e}, 2nd tail moment {tr['K20_m2']:.2e}\n")
        f.write(f"- Truncation K=60: mass {tr['K60_mass']:.2e}, 2nd tail moment {tr['K60_m2']:.2e}\n")
        f.write(f"- Density difference K20 vs K60: {tr['dmax']:.2e}\n")
        f.write(f"- h sensitivity (h=1e-3 vs 5e-4), max |Δf| at left edge: {tr['h_sens']:.3e} (edge sharpening; report only)\n")
        f.write(f"- Dual-solver density deviation (x>0.15): {s0['dual_solver_dev_pct']:.3f}%\n")
        f.write("- Zero-mass checks: all within 4e-8 of analytic formula.\n")
        f.write("- Companion route: auxiliary u = c·m, e = u + (c-1)/z; internal identity ∫1/(1+tu) = -z·e (see NOTE_伴随方程形式问题.md).\n")

    with open(os.path.join(OUT, "A_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    note("=== APPENDIX ITEMS DONE ===")

if __name__ == "__main__":
    main()
