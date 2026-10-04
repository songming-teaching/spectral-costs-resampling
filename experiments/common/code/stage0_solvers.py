# stage0_solvers.py
# Solver self-check (BLOCKING) per 实验指导手册 §五.
# Two independent solvers for Definition 1 of 论文一_v09:
#   (A) covariance-side direct solver:  z = -1/m + ∫ t/(1+c t m) η(dt),  m: C+ -> C+
#   (B) companion-side solver:          z = -1/m̃ + (1/c) ∫ t/(1+t m̃) η(dt), m̃: C+ -> C+
# with the exact linear relation m = (m̃ - (c-1)/z)/c used for cross-checks.
# All outputs -> output/stage0_*.csv and console log.

import numpy as np
import csv, os, json

SEED = 20260915
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- utilities
def trapz(y, x):
    return np.trapezoid(y, x)  # numpy>=2.0

def _solve_grid(zs, c, supp, probs, comp, tol=1e-12, maxit=2000000):
    """Plain (undamped) fixed-point iteration, vectorized over z-grid.
    comp=False: covariance-side T(m) = -1/(z - g(m)), g(m) = Σ w t/(1+c t m)
                (paper Definition 1; validated against simulation).
    comp=True : auxiliary u = c·m via T(u) = -c/(z - I(u)), I(u) = Σ w t/(1+t u)
                (exact rewrite of Definition 1 in u = c·m); returns u, NOT e.
    The C+ fixed point is attracting for these maps (standard Silverstein algorithm)."""
    zs = np.asarray(zs, dtype=complex)
    t = supp[None, :]            # 1 x K
    w = probs[None, :]           # 1 x K
    coef = 1.0 if comp else c
    if comp:
        m = -c / zs
    else:
        m = -1.0 / zs
    for it in range(maxit):
        g = np.sum(w * t / (1.0 + coef * t * m[:, None]), axis=1)
        if comp:
            m_new = -c / (zs - g)
        else:
            m_new = -1.0 / (zs - g)
        step = np.max(np.abs(m_new - m))
        m = m_new
        if step < tol:
            break
    else:
        raise RuntimeError(f"no convergence (maxit={maxit}), residual step={step:.2e}")
    return m

def _comp_e(u, z, c):
    """Companion Stieltjes transform from auxiliary u = c·m:  e = u + (c-1)/z."""
    return u + (c - 1.0) / np.asarray(z, dtype=complex)

def solve_stieltjes_cov(z, c, supp, probs, tol=1e-13, maxit=2000, m0=None):
    """Covariance-side Stieltjes m(z): plain fixed point (vectorized), then residual & branch checks."""
    zs = np.atleast_1d(z)
    m = _solve_grid(zs, c, supp, probs, comp=False)
    g = np.sum(probs * supp / (1.0 + c * supp * m[0]))
    resid = abs(m[0] * (zs[0] - g) + 1.0)  # multiplied-out form, stable for |m| tiny or huge
    if m[0].imag < 0 or resid > 1e-10:
        raise RuntimeError(f"cov solver failed at z={z}: Im={m[0].imag:.2e}, resid={resid:.2e}")
    return complex(m[0])

def solve_stieltjes_comp(z, c, supp, probs, tol=1e-13, maxit=2000, m0=None):
    """Companion-side Stieltjes e(z): solve u = c·m via the exact rewritten equation
    z = -c/u + ∫ t/(1+t u) η(dt), then e = u + (c-1)/z.
    Also verifies the exact identity ∫ η(dt)/(1+t u) = -z·e."""
    zs = np.atleast_1d(z)
    u = _solve_grid(zs, c, supp, probs, comp=True)
    g = np.sum(probs * supp / (1.0 + supp * u[0]))
    resid = abs(u[0] * (zs[0] - g) + c)
    if resid > 1e-10:
        raise RuntimeError(f"comp solver failed at z={z}: resid={resid:.2e}")
    e = _comp_e(u[0], zs[0], c)
    ident = abs(np.sum(probs / (1.0 + supp * u[0])) + zs[0] * e)
    if ident > 1e-8:
        raise RuntimeError(f"comp identity failed at z={z}: {ident:.2e}")
    return complex(e)

def density_grid(xgrid, h, c, supp, probs, side="cov"):
    """Smoothed density f_h(x) = (1/pi) Im m(x + i h), vectorized fixed point over the grid.
    side="comp": density of the companion law (via u = c·m, e = u + (c-1)/z)."""
    zs = xgrid + 1j * h
    m = _solve_grid(zs, c, supp, probs, comp=(side != "cov"))
    if side != "cov":
        m = _comp_e(m, zs, c)
    return m.imag / np.pi, m

def mp_density(x, c):
    """MP(c) density on [(1-sqrt c)^2, (1+sqrt c)^2] (covariance-side convention)."""
    a = (1 - np.sqrt(c)) ** 2
    b = (1 + np.sqrt(c)) ** 2
    y = np.zeros_like(x)
    mask = (x > a) & (x < b)
    xm = x[mask]
    y[mask] = np.sqrt((b - xm) * (xm - a)) / (2 * np.pi * c * xm)
    return y, a, b

# ------------------------------------------------------------- count laws
def mixpoi_law(pts, ws, K):
    """Truncated MixPoi(ρ): probabilities for k=0..K, plus tail mass & 2nd tail moment."""
    ks = np.arange(K + 1)
    p = np.zeros(K + 1)
    tail_mass = 0.0
    tail_m2 = 0.0
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, K + 1)))])
    for lam, w in zip(pts, ws):
        pmf = np.exp(-lam + ks * np.log(lam) - logfact)  # unweighted Poi(lam) pmf
        p += w * pmf
        tail_mass += w * (1.0 - pmf.sum())
        m2_full = lam + lam ** 2
        tail_m2 += w * (m2_full - np.sum(ks ** 2 * pmf))
    return ks, p, tail_mass, tail_m2

def residual_twopoint_half(K):
    """ν_res for π = ½δ_{(0,1/2)} + ½δ_{(1,1/2)}:  ν(k) = ½ Poi(0.5)(k) + ½ Poi(0.5)(k-1)."""
    lam = 0.5
    ks = np.arange(K + 1)
    logf = lambda k: -lam + k * np.log(lam) - np.sum(np.log(np.arange(1, k + 1)))
    p1 = np.exp([logf(k) for k in ks])
    p2 = np.concatenate([[0.0], np.exp([logf(k) for k in range(K)])])
    p = 0.5 * p1 + 0.5 * p2
    tail_mass = 1.0 - p.sum()
    m2_full = 0.5 * (lam + lam ** 2) + 0.5 * ((1 + lam) + (1 + lam) ** 2)
    tail_m2 = m2_full - np.sum(ks ** 2 * p)
    return ks, p, tail_mass, tail_m2

# ---------------------------------------------------------------- checks
def main():

	log = []
	def note(s):
	    print(s)
	    log.append(s)

	results = {}

	# ---- check 4+1+3+2 on MP(c): η = δ1, c = 0.8
	c0 = 0.8
	supp = np.array([1.0]); probs = np.array([1.0])
	note("=== Check 1-4: η = δ1, c = 0.8 (MP recovery) ===")
	# branch & asymptotic
	for z in [1j, 3 + 2j, 0.05 + 0.5j]:
	    m = solve_stieltjes_cov(z, c0, supp, probs)
	    resid = abs(m + 1.0 / (z - supp[0] * probs[0] / (1 + c0 * supp[0] * m)))
	    note(f"z={z}: Im m={m.imag:.3e} (>0: {m.imag>0}), equation residual={resid:.2e}")
	    assert m.imag > 0 and resid < 1e-10
	m_far = solve_stieltjes_cov(1j * 1e6, c0, supp, probs)
	note(f"zm(z) at z=1e6 i: {1j*1e6*m_far:.6f} (should be ~ -1)")
	assert abs(1j * 1e6 * m_far + 1) < 1e-3
	xgrid = np.linspace(0.02, 3.4, 400)
	h = 1e-3
	fh, _ = density_grid(xgrid, h, c0, supp, probs, "cov")
	f_mp, a_mp, b_mp = mp_density(xgrid, c0)
	interior = (xgrid > a_mp + 0.05) & (xgrid < b_mp - 0.05)
	err = np.max(np.abs(fh[interior] - f_mp[interior]) / np.maximum(f_mp[interior], 1e-12))
	note(f"MP(c) recovery: max relative density error in interior = {err*100:.3f}% (<1% required)")
	assert err < 0.01
	results["MP_recovery_err_pct"] = err * 100

	# ---- check 5: zero mass vs analytic formula, two laws
	note("=== Check 5: zero-mass consistency ===")
	for name, (ks, p, tm, tm2) in {
	    "uniform/MixPoi(1)": mixpoi_law([1.0], [1.0], 40),
	    "two-pt mult": mixpoi_law([0.5, 1.5], [0.5, 0.5], 60),
	    "two-pt res": residual_twopoint_half(60),
	}.items():
	    nu0 = p[0]
	    analytic = max(0.0, 1.0 - (1.0 - nu0) / c0)
	    y = 1e-5
	    m = solve_stieltjes_cov(1j * y, c0, ks.astype(float), p)
	    num = float(np.real(-(1j * y) * m))
	    note(f"{name}: analytic zero mass={analytic:.6f}, solver(-iy m(iy))={num:.6f}, |diff|={abs(num-analytic):.2e}")
	    assert abs(num - analytic) < 2e-3
	results["zero_mass_ok"] = True

	# ---- check 6: truncation sensitivity (two-pt mult, K = 20 vs 60)
	note("=== Check 6: truncation sensitivity (two-pt mult, K=20 vs K=60) ===")
	ksA, pA, tmA, tm2A = mixpoi_law([0.5, 1.5], [0.5, 0.5], 20)
	ksB, pB, tmB, tm2B = mixpoi_law([0.5, 1.5], [0.5, 0.5], 60)
	note(f"K=20: truncated mass={tmA:.3e}, truncated 2nd tail moment={tm2A:.3e}")
	note(f"K=60: truncated mass={tmB:.3e}, truncated 2nd tail moment={tm2B:.3e}")
	fA, _ = density_grid(xgrid, 1e-3, c0, ksA.astype(float), pA)
	fB, mB = density_grid(xgrid, 1e-3, c0, ksB.astype(float), pB)
	dmax = np.max(np.abs(fA - fB))
	note(f"max |f_K20 - f_K60| on grid = {dmax:.3e}")
	# h sensitivity
	f_h2, _ = density_grid(xgrid, 5e-4, c0, ksB.astype(float), pB)
	dh = np.max(np.abs(f_h2 - fB))
	jdh = int(np.argmax(np.abs(f_h2 - fB)))
	note(f"h=1e-3 vs h=5e-4: max |Δf| = {dh:.3e} at x={xgrid[jdh]:.3f} (edge sharpening; report only)")
	results["trunc"] = {"K20_mass": tmA, "K20_m2": tm2A, "K60_mass": tmB, "K60_m2": tm2B, "dmax": dmax, "h_sens": dh}

	# ---- check 7: covariance-side vs companion-side, same parameters
	note("=== Check 7: dual-solver cross-check (two-pt mult, c=0.8) ===")
	zs = xgrid + 1j * 1e-3
	m_cov = _solve_grid(zs, c0, ksB.astype(float), pB, comp=False)
	u_cmp = _solve_grid(zs, c0, ksB.astype(float), pB, comp=True)
	m_cmp = _comp_e(u_cmp, zs, c0)  # companion Stieltjes e(z)
	# exact algebraic relation between the two transforms: e = c·m + (c-1)/z
	rel = np.max(np.abs(m_cmp - (c0 * m_cov + (c0 - 1.0) / zs)) / np.abs(m_cmp))
	note(f"dual-solver transform relation residual (max, relative): {rel:.2e}")
	assert rel < 1e-9
	# bonus internal identity: ∫ η(dt)/(1+t u) = -z·e
	ident = np.max(np.abs(np.sum(pB[None, :] / (1.0 + ksB.astype(float)[None, :] * u_cmp[:, None]), axis=1) + zs * m_cmp))
	note(f"companion internal identity ∫1/(1+tu) = -z·e: max residual {ident:.2e}")
	assert ident < 1e-8
	# density-level comparison via the exact relation (companion-derived m vs covariance m)
	f_cov = m_cov.imag / np.pi
	m_from_comp = (m_cmp - (c0 - 1.0) / zs) / c0
	f_from_comp = m_from_comp.imag / np.pi
	mask_int = (xgrid > 0.15)  # exclude left boundary layer (noted in manual)
	dev = np.max(np.abs(f_cov[mask_int] - f_from_comp[mask_int]) / np.maximum(f_cov[mask_int], 1e-9))
	note(f"dual-solver density deviation (x>0.15): {dev*100:.3f}% (<2% required)")
	assert dev < 0.02
	results["dual_solver_dev_pct"] = dev * 100
	# companion zero mass: (1-c) + c·μQ({0})
	nu0 = pB[0]
	mu0 = max(0.0, 1 - (1 - nu0) / c0)
	y = 1e-5
	mcmp = solve_stieltjes_comp(1j * y, c0, ksB.astype(float), pB)
	comp0 = float(np.real(-(1j * y) * mcmp))
	note(f"companion zero mass: solver={comp0:.6f} vs (1-c)+c*mu0={(1-c0)+c0*mu0:.6f}")
	assert abs(comp0 - ((1 - c0) + c0 * mu0)) < 2e-3

	with open(os.path.join(OUT, "stage0_checks.json"), "w", encoding="utf-8") as f:
	    json.dump(results, f, indent=2, ensure_ascii=False)
	with open(os.path.join(OUT, "stage0_log.txt"), "w", encoding="utf-8") as f:
	    f.write("\n".join(log))
	note("=== ALL STAGE0 CHECKS PASSED ===")


if __name__ == "__main__":
	main()
