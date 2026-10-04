# -*- coding: utf-8 -*-
"""
plot_panels.py —— 把 output/ 下各多联图（multi-panel figure）拆成独立的单 panel 矢量图。

【不改动任何数据】的保证方式
  1) E2 / E4_counts 两类 panel 的数据本来就落盘在 CSV 里 —— 直接读 CSV，不重算。
  2) E1 / E3 / E4_spectral / A3 的绘图数据原脚本只在内存里生成、从未落盘。
     这里用与原脚本 **完全相同的 SEED(20260915) 与完全相同的随机数消费顺序** 精确重放，
     复现出与原图逐点相同的数据；随后用已存盘 CSV 中的统计量对重放结果做逐项断言校验。
     校验通过即证明随机数流与原运行完全一致（因而绘图数据也完全一致）。
     校验明细写入 output/panels/REPLAY_VERIFICATION.txt。
  3) 原脚本、原 CSV、原 PNG 一律不修改、不覆盖。新图只写到 output/panels/。

【出版规格】（按需求）
  * 1:1 最终尺寸绘制，宽度 3.4 in（IEEE 单栏）；高度按 panel 内容取 2.3–2.7 in。
  * 字体族统一 Times New Roman；数学符号用 mathtext + STIX（Times 兼容字形），
    避免原脚本里的 Unicode 上标/特殊符号（⁻¹ ⁺ ⊠ √ −）在 Times New Roman 下缺字形变成方框。
  * 基础字号 9 pt；图例/坐标轴标签 9 pt；刻度数字 8 pt —— 全部 >= 8 pt 下限。
  * 刻度：tick_params(direction='in', top=True, right=True)，四边刻度朝内。
  * 矢量 PDF，保存时 bbox_inches='tight' 去除四周白边；pdf.fonttype=42（文字可在 AI 中编辑）。

用法
  python plot_panels.py                    # 全部宽度预设（single/span2/span3/col2）
  python plot_panels.py --width single      # 只出单栏 3.4 in
  python plot_panels.py --width span3       # 只出跨栏三联位 2.33 in
  python plot_panels.py --width 2.6         # 任意英寸数值
  python plot_panels.py --recompute         # 强制重新重放（忽略缓存）
  python plot_panels.py --only E1           # 只输出名字含 "E1" 的 panel

  每个宽度输出到 output/panels/w<宽度>_<预设名>/，互不覆盖。
  子图按最终印出宽度绘制 —— 插入 LaTeX 时不要再用 width= 缩放，否则字号跟着缩。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
PANEL_DIR = os.path.join(OUT, "panels")
CACHE_NPZ = os.path.join(PANEL_DIR, "_replay_cache.npz")
CACHE_META = os.path.join(PANEL_DIR, "_replay_meta.json")
VERIFY_TXT = os.path.join(PANEL_DIR, "REPLAY_VERIFICATION.txt")

sys.path.insert(0, HERE)
import stage0_solvers as S0          # 有 __main__ 守卫，导入无副作用
import stage3_appendix as A3mod      # 有 __main__ 守卫，导入无副作用

SEED = 20260915

# ------------------------------------------------------------------ 出版规格常量
FONT_FAMILY = "Times New Roman"
BASE_FS = 9.0        # 基础字号（标题/坐标轴标签/图例）
MIN_FS = 8.0         # 最小字号（刻度数字）—— 不得低于此值
PANEL_W = 3.4        # in，当前绘制宽度（由 --width 设定，运行时可变）
PAD_IN = 0.02        # bbox_inches='tight' 的额外留白

# IEEE TSP 版面：单栏 3.4 in，跨栏 7.16 in。
# 关键约束：子图必须按【最终印出宽度】绘制，插入 LaTeX 时不再用 width= 缩放，
# 否则字号随图一起缩小，等于回到多联图的老问题。
# 各预设对应的 LaTeX 落位见 PANELS_README.md。
WIDTH_PRESETS = {
    "single": 3.40,   # 单栏，一图独占        \includegraphics{...}
    "span2":  3.50,   # 跨栏 figure*，两图并排  7.16/2 − 间距
    "span3":  2.33,   # 跨栏 figure*，三图并排  7.16/3 − 间距
    "col2":   1.66,   # 单栏内两图并排（很窄，仅在标签极短时可用）
}
NARROW_TH = 2.80     # 低于此宽度启用紧凑标签（短标题、短图例）


def narrow() -> bool:
    return PANEL_W < NARROW_TH


def lab(long: str, short: str) -> str:
    """按当前 panel 宽度选择长/短标签。窄图不缩字号，只缩文字。"""
    return short if narrow() else long


def panel_dir() -> str:
    """每个宽度一个子目录，互不覆盖。"""
    tag = next((k for k, v in WIDTH_PRESETS.items() if abs(v - PANEL_W) < 1e-9), None)
    sub = f"w{PANEL_W:.2f}" + (f"_{tag}" if tag else "")
    return os.path.join(PANEL_DIR, sub)


def apply_ieee_style() -> None:
    rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT_FAMILY, "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": BASE_FS,
        "axes.titlesize": BASE_FS,
        "axes.labelsize": BASE_FS,
        "xtick.labelsize": MIN_FS,
        "ytick.labelsize": MIN_FS,
        "legend.fontsize": BASE_FS,
        "mathtext.fontset": "stix",          # 与 Times 观感一致的数学字形
        "axes.unicode_minus": False,
        "axes.linewidth": 0.6,
        "axes.labelpad": 2.0,
        "axes.edgecolor": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.2,
        "patch.linewidth": 0.5,
        # 不设 savefig.bbox="tight"：由 constrained layout 保证 1:1 尺寸，
        # tight 会按内容重新裁剪／外扩画布，破坏目标宽度。
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def new_panel(h: float = 2.5):
    """按最终尺寸 1:1 建一个单 panel 画布；四边刻度朝内。

    h 是 3.4 in 基准宽度下的高度；窄 panel 等比缩放（保留 0.86 下限，
    避免过扁导致 y 轴刻度挤在一起）。
    """
    scale = max(0.86, PANEL_W / 3.40)
    # constrained layout：在给定 figsize 内压缩坐标区以容纳标签，
    # 输出宽度严格等于 PANEL_W。若改用 bbox_inches="tight"，画布会为了
    # 容纳长标签而向外扩张，输出就会超过目标宽度（窄图尤其明显）。
    fig, ax = plt.subplots(figsize=(PANEL_W, h * scale), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=MIN_FS, width=0.6, length=3.0)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    return fig, ax


def compact_legend(ax, **kw):
    d = dict(fontsize=BASE_FS, frameon=True, framealpha=0.92, edgecolor="0.35",
             borderpad=0.32, labelspacing=0.28,
             handlelength=1.6, handletextpad=0.45)
    d.update(kw)
    leg = ax.legend(**d)
    leg.get_frame().set_linewidth(0.5)
    return leg


SAVED: list[tuple[str, float, float]] = []


def save_panel(fig, name: str) -> str:
    """保存为矢量 PDF（tight 裁白边），并记录裁完后的真实尺寸。"""
    d = panel_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".pdf")
    fig.canvas.draw()
    w_in, h_in = (float(v) for v in fig.get_size_inches())
    # 不用 bbox_inches="tight"：constrained layout 已把所有元素收进 figsize，
    # 再 tight 一次会重新按内容裁剪／外扩，破坏 1:1 尺寸保证。
    fig.savefig(path, format="pdf")
    plt.close(fig)
    SAVED.append((name, w_in, h_in))
    if w_in > PANEL_W + 0.02:
        print(f"  !! 警告：{name} tight 后宽度 {w_in:.3f} in 超过 {PANEL_W} in（可能有元素突出画布）")
    return path


def read_csv_dicts(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =========================================================================
# 第一部分：随机数流精确重放（复现原脚本的绘图数据）
# =========================================================================
def _e1_spectra_and_counts(lam, rng):
    """逐行对应 stage2_E1.py 的 spectra_and_counts()，随机数消费顺序与数值路径完全一致。"""
    N, d = 1000, 800
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


def replay_E1():
    rng = np.random.default_rng(SEED)
    c, N, R, K = 0.8, 1000, 20, 60
    xgrid = np.linspace(0.05, 6.0, 300)
    h_plot = 1e-3
    zgrid_Eh = np.linspace(0.05, 8.0, 40) + 1j * 0.05
    z2 = np.linspace(0.05, 8.0, 40) + 1j * 0.025

    ksU, pU_mult, _, _ = S0.mixpoi_law([1.0], [1.0], K)
    pU_res = (np.arange(K + 1) == 1).astype(float)
    ks2, p2_mult, _, _ = S0.mixpoi_law([0.5, 1.5], [0.5, 0.5], K)
    ks2r, p2_res, _, _ = S0.residual_twopoint_half(K)

    PROFILES = {
        "uniform": dict(
            lam=np.ones(N),
            laws={"multinomial": (ksU.astype(float), pU_mult),
                  "residual": (ksU.astype(float), pU_res)},
            supp_pre=np.array([1.0]), p_pre=np.array([1.0]),
            preg=dict(m2_pre=1.8, m2_mult=2.6, m2_res=1.8,
                      nu0_mult=0.367879, nu0_res=0.0, mu0_mult=0.20985, mu0_res=0.0),
        ),
        "twopoint": dict(
            lam=np.concatenate([np.full(N // 2, 0.5), np.full(N - N // 2, 1.5)]),
            laws={"multinomial": (ks2.astype(float), p2_mult),
                  "residual": (ks2r.astype(float), p2_res)},
            supp_pre=np.array([0.5, 1.5]), p_pre=np.array([0.5, 0.5]),
            preg=dict(m2_pre=2.0, m2_mult=2.8, m2_res=2.4,
                      nu0_mult=0.41483, nu0_res=0.30327, mu0_mult=0.26854, mu0_res=0.12909),
        ),
    }

    # ---- 第一轮循环：统计量（用于对齐 E1_moments.csv / E1_stieltjes.csv，从而证明随机数流一致）
    mom_rows, sth_rows, curves = [], [], {}
    for pname, prof in PROFILES.items():
        lam, preg = prof["lam"], prof["preg"]
        reps = [_e1_spectra_and_counts(lam, rng) for _ in range(R)]
        cur = {"pre": S0.density_grid(xgrid, h_plot, c, prof["supp_pre"], prof["p_pre"])[0]}
        for scheme in ["multinomial", "residual"]:
            sp, pb = prof["laws"][scheme]
            cur[scheme] = S0.density_grid(xgrid, h_plot, c, sp, pb)[0]
        curves[pname] = cur

        for scheme in ["pre", "multinomial", "residual"]:
            m1s = np.array([r[scheme]["m1"] for r in reps])
            m2s = np.array([r[scheme]["m2"] for r in reps])
            nu0s = np.array([r[scheme]["nu0"] for r in reps])
            mu0s = np.array([r[scheme]["mu0"] for r in reps])
            if scheme == "pre":
                m2_th, nu0_th, mu0_th = preg["m2_pre"], 0.0, 0.0
            else:
                tag = "mult" if scheme == "multinomial" else "res"
                m2_th, nu0_th, mu0_th = preg[f"m2_{tag}"], preg[f"nu0_{tag}"], preg[f"mu0_{tag}"]
            mom_rows.append(dict(profile=pname, scheme=scheme,
                                 m1_emp=m1s.mean(), m1_se=m1s.std(ddof=1) / np.sqrt(R),
                                 m2_emp=m2s.mean(), m2_se=m2s.std(ddof=1) / np.sqrt(R),
                                 nu0_emp=nu0s.mean(), nu0_se=nu0s.std(ddof=1) / np.sqrt(R),
                                 mu0_emp=mu0s.mean(), mu0_se=mu0s.std(ddof=1) / np.sqrt(R),
                                 m2_th=m2_th, nu0_th=nu0_th, mu0_th=mu0_th))
        for scheme in ["pre", "multinomial", "residual"]:
            pooled = np.concatenate([r[scheme]["eigs"] for r in reps])
            sp, pb = (prof["supp_pre"], prof["p_pre"]) if scheme == "pre" else prof["laws"][scheme]
            m_emp = np.mean(1.0 / (pooled[None, :] - zgrid_Eh[:, None]), axis=1)
            m_th = S0._solve_grid(zgrid_Eh, c, sp, pb, comp=False)
            m_emp2 = np.mean(1.0 / (pooled[None, :] - z2[:, None]), axis=1)
            m_th2 = S0._solve_grid(z2, c, sp, pb, comp=False)
            sth_rows.append(dict(profile=pname, scheme=scheme,
                                 Eh_max=float(np.max(np.abs(m_emp - m_th))),
                                 Eh_mean=float(np.mean(np.abs(m_emp - m_th))),
                                 Eh_h2_max=float(np.max(np.abs(m_emp2 - m_th2)))))

    # ---- 第二轮循环：原脚本真正用于画图的数据（fresh pooled for plots）
    plot = {}
    for pname, prof in PROFILES.items():
        reps_plot = [_e1_spectra_and_counts(prof["lam"], rng) for _ in range(R)]
        plot[pname] = {}
        for scheme in ["pre", "multinomial", "residual"]:
            e = np.concatenate([r[scheme]["eigs"] for r in reps_plot])
            plot[pname][scheme] = dict(
                n_pooled=np.concatenate([r[scheme]["n"] for r in reps_plot]),
                eigs_pooled=e,
                mu0_emp_plot=float(np.mean(e <= 1e-10)),
            )

    arr = {"E1__xgrid": xgrid}
    for pname in PROFILES:
        for scheme in ["pre", "multinomial", "residual"]:
            arr[f"E1__{pname}__{scheme}__n"] = plot[pname][scheme]["n_pooled"]
            arr[f"E1__{pname}__{scheme}__eigs"] = plot[pname][scheme]["eigs_pooled"]
            arr[f"E1__{pname}__{scheme}__mu0_emp_plot"] = np.array([plot[pname][scheme]["mu0_emp_plot"]])
            arr[f"E1__{pname}__curve__{scheme}"] = curves[pname][scheme]
            preg = PROFILES[pname]["preg"]
            mu0_th = 0.0 if scheme == "pre" else preg["mu0_" + ("mult" if scheme == "multinomial" else "res")]
            arr[f"E1__{pname}__{scheme}__mu0_th"] = np.array([mu0_th])
            if scheme != "pre":
                ks_th, p_th = PROFILES[pname]["laws"][scheme]
                arr[f"E1__{pname}__{scheme}__ks_th"] = np.asarray(ks_th, dtype=float)
                arr[f"E1__{pname}__{scheme}__p_th"] = np.asarray(p_th, dtype=float)
    mom_keys = ["m1_emp", "m1_se", "m2_emp", "m2_se", "nu0_emp", "nu0_se", "mu0_emp", "mu0_se",
                "m2_th", "nu0_th", "mu0_th"]
    arr["E1__mom"] = np.array([[r[k] for k in mom_keys] for r in mom_rows])
    arr["E1__sth"] = np.array([[r["Eh_max"], r["Eh_mean"], r["Eh_h2_max"]] for r in sth_rows])
    meta = {"E1__mom_labels": [[r["profile"], r["scheme"]] for r in mom_rows],
            "E1__mom_keys": mom_keys,
            "E1__sth_labels": [[r["profile"], r["scheme"]] for r in sth_rows],
            "E1__sth_keys": ["Eh_max", "Eh_mean", "Eh_h2_max"]}
    return arr, meta


def replay_E3():
    rng = np.random.default_rng(SEED)
    c, N, d, R, K = 0.8, 1000, 800, 30, 60
    PROFILES = {
        "A": dict(lam=np.concatenate([np.full(500, 0.25), np.full(500, 1.75)]),
                  pts=[0.25, 1.75], ws=[0.5, 0.5],
                  preg=dict(m2=3.05, nu0=0.476287, mu0=0.345359, b3=8.375)),
        "B": dict(lam=np.concatenate([np.full(900, 0.75), np.full(100, 3.25)]),
                  pts=[0.75, 3.25], ws=[0.9, 0.1],
                  preg=dict(m2=3.05, nu0=0.429007, mu0=0.286259, b3=9.5)),
    }
    rows, eigs_pool, ess = [], {}, {}
    for name, prof in PROFILES.items():
        lam = prof["lam"]
        w = lam / N
        e = N ** 2 / float(np.sum(lam ** 2))
        ess[name] = e
        m2s, nu0s, mu0s, m3s, eall = [], [], [], [], []
        for r in range(R):
            X = rng.standard_normal((d, N))
            n = rng.multinomial(N, w).astype(float)
            S = (X * n) @ X.T / N
            eigs = np.linalg.eigvalsh(S)
            eall.append(eigs)
            M = int(np.count_nonzero(n))
            m2s.append(float((eigs ** 2).mean()))
            m3s.append(float((eigs ** 3).mean()))
            nu0s.append(float(np.mean(n == 0)))
            mu0s.append((d - min(d, M)) / d)
        eigs_pool[name] = np.concatenate(eall)
        pr = prof["preg"]
        m3_th = 1 + 3 * c * (1 + 25 / 16) + c ** 2 * pr["b3"]
        f = lambda a: (np.mean(a), np.std(a, ddof=1) / np.sqrt(R))
        rows.append(dict(profile=name, ess=e,
                         m2_emp=f(m2s)[0], m2_se=f(m2s)[1], m2_th=pr["m2"],
                         nu0_emp=f(nu0s)[0], nu0_se=f(nu0s)[1], nu0_th=pr["nu0"],
                         mu0_emp=f(mu0s)[0], mu0_se=f(mu0s)[1], mu0_th=pr["mu0"],
                         m3_emp=f(m3s)[0], m3_se=f(m3s)[1], m3_th=m3_th))
    keys = ["ess", "m2_emp", "m2_se", "m2_th", "nu0_emp", "nu0_se", "nu0_th",
            "mu0_emp", "mu0_se", "mu0_th", "m3_emp", "m3_se", "m3_th"]
    arr = {"E3__eigs__A": eigs_pool["A"], "E3__eigs__B": eigs_pool["B"],
           "E3__rows": np.array([[r[k] for k in keys] for r in rows])}
    meta = {"E3__labels": [r["profile"] for r in rows], "E3__keys": keys}
    return arr, meta


def replay_E4():
    rng = np.random.default_rng(SEED)
    ALPHA, REPS = 0.4, 200
    Ns = [2 ** m for m in range(8, 15)]

    def build(construction, N):
        if construction == "uniform":
            lam = np.ones(N)
        elif construction == "interior":
            aN = int(round(ALPHA * N))
            aN = min(max(aN, 1), N - 1)
            eps = N ** -0.5
            a = aN / N
            lam = np.concatenate([np.full(aN, 1.0 - eps),
                                  np.full(N - aN, 1.0 + (a / (1.0 - a)) * eps)])
        elif construction == "endpoint":
            lam = np.concatenate([np.full(N - 1, 1.0 - 1.0 / N), np.array([2.0 - 1.0 / N])])
        else:
            raise ValueError(construction)
        k = np.floor(lam).astype(np.int64)
        u = lam - k
        R = int(N - k.sum())
        return lam, k, u, R

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
            return (R / N, lam2_dev, lam2_mean, zfrac.mean(), zfrac.std(ddof=1) / np.sqrt(reps),
                    m2.mean(), m2.std(ddof=1) / np.sqrt(reps), summary)
        n = k
        return (R / N, lam2_dev, lam2_mean, float(np.mean(n == 0)), 0.0,
                float(np.mean(n.astype(float) ** 2)), 0.0, summary)

    rows = []
    for cons in ["uniform", "interior", "endpoint"]:
        for N in Ns:
            RnN, dev2, lam2m, z, zse, m2, m2se, summ = run(cons, N, REPS)
            rows.append(dict(cons=cons, N=N, RnN=RnN, dev2=dev2, lam2_mean=lam2m,
                             zero=z, zero_se=zse, m2=m2, m2_se=m2se,
                             lam_min=summ[0], lam_max=summ[1], lam_maxdev=summ[2]))

    # ---- spectral panel（原 stage1_E4.py 第 130–160 行，随机数流紧接上面的 counts 循环）
    A2 = 0.5
    N, d = 1024, 819
    c = d / N
    aN = int(round(A2 * N))
    a = aN / N
    eps = N ** -0.5
    lam = np.concatenate([np.full(aN, 1.0 - eps), np.full(N - aN, 1.0 + (a / (1 - a)) * eps)])
    k = np.floor(lam).astype(np.int64)
    u = lam - k
    R = int(N - k.sum())
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

    K = 60
    ks_poi, p_poi, _, _ = S0.mixpoi_law([1.0], [1.0], K)
    ks_res = np.arange(K + 1)
    p_res = 0.5 * (ks_res == 1) + 0.5 * p_poi
    # 理论曲线的求值网格。原为 0.02..4.0，但经验谱延伸到 ~6.7（Prop. 13：
    # multinomial 的极限右支撑无界），拆成独立子图并统一 x 轴后，红线在 4 处
    # 断掉会被误读为理论失配。扩到 7.0 覆盖整个显示窗口。纯显示网格，不消耗随机数。
    xgrid = np.linspace(0.02, 7.0, 700)
    # pre 面板的理论曲线是 T_c(δ₁) = MP(c)：supp 与 probs 都必须是单点 [1.0]。
    # 原脚本 stage1_E4.py 写作 supp=ks_poi*0+1.0（61 个 1.0）而 probs=[1.0]（1 个），
    # 长度不匹配导致返回的密度恒为 0 —— 原 E4_spectral.png 里 pre 面板那条红线
    # 是贴着横轴的平线，而非 MP(c)。此处修正。
    f_pre, _ = S0.density_grid(xgrid, 1e-3, c, np.array([1.0]), np.array([1.0]))
    f_mult, _ = S0.density_grid(xgrid, 1e-3, c, ks_poi.astype(float), p_poi)
    f_res, _ = S0.density_grid(xgrid, 1e-3, c, ks_res.astype(float), p_res)
    nu0_mult = p_poi[0]
    nu0_res = p_res[0]
    mu0_mult = max(0, 1 - (1 - nu0_mult) / c)
    mu0_res = max(0, 1 - (1 - nu0_res) / c)
    mu0_pre = max(0, 1 - 1 / c)

    keys = ["RnN", "dev2", "lam2_mean", "zero", "zero_se", "m2", "m2_se",
            "lam_min", "lam_max", "lam_maxdev"]
    arr = {
        "E4__rows": np.array([[r[k] for k in keys] for r in rows]),
        "E4__xgrid": xgrid, "E4__f_pre": f_pre, "E4__f_mult": f_mult, "E4__f_res": f_res,
        "E4__mu0": np.array([mu0_pre, mu0_mult, mu0_res]),
        "E4__params": np.array([N, d, c]),
        **{f"E4__spec__{s}": spectra[s] for s in spectra},
    }
    meta = {"E4__labels": [[r["cons"], r["N"]] for r in rows], "E4__keys": keys}
    return arr, meta


def replay_A3():
    rng = np.random.default_rng(SEED)
    c = A3mod.c

    # ---- control（随机数必须消费；eigs 用于 control_max_deviation 校验）
    d_c, N_c = 2000, 2500
    sig_c = np.concatenate([np.full(d_c // 2, 0.5), np.full(d_c - d_c // 2, 1.5)])
    Z_c = rng.standard_normal((d_c, N_c))
    S_c = np.diag(np.sqrt(sig_c)) @ (Z_c @ Z_c.T / N_c) @ np.diag(np.sqrt(sig_c))
    eigs_c = np.linalg.eigvalsh(S_c)
    devs = []
    for z in [1j, 0.5 + 0.5j, 2 + 1j, 0.2 + 0.3j]:
        m_th, _ = A3mod.m_product(z, A3mod.mfun_H, A3mod.mfun_MP)
        m_emp = np.mean(1.0 / (eigs_c - z))
        devs.append(abs(m_th - m_emp))
    control_max = float(max(devs))

    # ---- A1
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
            # 不出图：m2 = mean(eigs²) = tr(S²)/d = ‖S‖²_F/d（S 实对称），免特征分解。
            # 随机数消费与原 stage3_appendix.py 完全一致。
            M = int(np.count_nonzero(n))
            m2s.append(float(np.sum(S * S) / d_v))
            nu0s.append(float(np.mean(n == 0)))
            mu0s.append((d_v - min(d_v, M)) / d_v)
        return dict(N=Nv, profile=pname, d=d_v,
                    m2=np.mean(m2s), m2_se=np.std(m2s, ddof=1) / 3.162, m2_th=th["m2_mult"],
                    nu0=np.mean(nu0s), nu0_se=np.std(nu0s, ddof=1) / 3.162, nu0_th=th["nu0"],
                    mu0=np.mean(mu0s), mu0_se=np.std(mu0s, ddof=1) / 3.162, mu0_th=th["mu0"])

    a1_rows = [one_scale(Nv, pn) for Nv in [500, 2000] for pn in ["uniform", "twopoint"]]

    # ---- A2
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
            # 不出图：m2 用 ‖S‖²_F/d 等价替代，随机数流不变。
            M = int(np.count_nonzero(n))
            m2s.append(float(np.sum(S * S) / d2))
            mu0s.append((d2 - min(d2, M)) / d2)
        m2_th_asym = 1 + c * 2.25
        m2_th_fin = 1 + c * 1.25 + (1 - np.sum((lam2 / N2) ** 2)) * (d2 + 1 + kap4) / N2
        a2_rows.append(dict(dist=dist, kappa4=kap4,
                            m2=np.mean(m2s), m2_se=np.std(m2s, ddof=1) / np.sqrt(R2),
                            m2_th_asymptotic=m2_th_asym, m2_th_finiteN=m2_th_fin,
                            mu0=np.mean(mu0s), mu0_se=np.std(mu0s, ddof=1) / np.sqrt(R2),
                            mu0_th=0.26854))

    # ---- A3（绘图数据本体）
    N3, d3, R3 = 1000, 800, 10
    sig3 = np.concatenate([np.full(d3 // 2, 0.5), np.full(d3 - d3 // 2, 1.5)])
    a1_H, m2H = 1.0, 1.25
    m2_pre_th = m2H + c * a1_H ** 2 * 1.0
    m2_post_th = m2H + c * a1_H ** 2 * 2.0
    m1s, m2s_pre, m2s_post, eigs_pool = [], [], [], []
    Sg = np.diag(np.sqrt(sig3))
    for r in range(R3):
        Z = rng.standard_normal((d3, N3))
        n = rng.multinomial(N3, np.full(N3, 1.0 / N3)).astype(float)
        e_post = np.linalg.eigvalsh(Sg @ ((Z * n) @ Z.T / N3) @ Sg)
        S_pre = Sg @ (Z @ Z.T / N3) @ Sg          # 实对称；m2_pre = ‖S_pre‖²_F/d3
        eigs_pool.append(e_post)
        m1s.append(e_post.mean())
        m2s_post.append((e_post ** 2).mean())
        m2s_pre.append(float(np.sum(S_pre * S_pre) / d3))
    eigs_pool = np.concatenate(eigs_pool)
    mu0_emp = float(np.mean(eigs_pool <= 1e-10))

    xgrid = np.linspace(0.02, 6.0, 250)
    f_prod = np.empty_like(xgrid)
    y_prev = None
    for j in range(len(xgrid) - 1, -1, -1):
        x = xgrid[j]
        mval, y_prev = A3mod.m_product(x + 1j * 2e-3, A3mod.mfun_H, A3mod.mfun_muQ,
                                       seed=None if y_prev is None else [y_prev.real, y_prev.imag])
        f_prod[j] = mval.imag / np.pi

    a1_keys = ["N", "d", "m2", "m2_se", "m2_th", "nu0", "nu0_se", "nu0_th", "mu0", "mu0_se", "mu0_th"]
    a2_keys = ["kappa4", "m2", "m2_se", "m2_th_asymptotic", "m2_th_finiteN", "mu0", "mu0_se", "mu0_th"]
    arr = {
        "A3__eigs_pool": eigs_pool, "A3__xgrid": xgrid, "A3__f_prod": f_prod,
        "A3__scalars": np.array([float(np.mean(m1s)), float(np.mean(m2s_pre)),
                                 float(np.mean(m2s_post)), mu0_emp, control_max,
                                 m2_pre_th, m2_post_th]),
        "A1__rows": np.array([[r[k] for k in a1_keys] for r in a1_rows]),
        "A2__rows": np.array([[r[k] for k in a2_keys] for r in a2_rows]),
    }
    meta = {"A1__labels": [[r["N"], r["profile"]] for r in a1_rows], "A1__keys": a1_keys,
            "A2__labels": [r["dist"] for r in a2_rows], "A2__keys": a2_keys}
    return arr, meta


# =========================================================================
# 第二部分：用已存盘 CSV 校验重放结果（证明数据未被更改）
# =========================================================================
CHECKS: list[dict] = []


def _chk(name, got, exp_raw, atol=1e-11, rtol=1e-9):
    exp = float(exp_raw)
    dev = abs(float(got) - exp)
    tol = atol + rtol * abs(exp)
    ok = bool(dev <= tol)
    CHECKS.append(dict(item=name, replayed=float(got), on_disk=exp, abs_dev=dev, tol=tol, ok=ok))
    return ok


def verify_against_csvs(arr, meta) -> None:
    """逐项对齐 output/ 下已存盘 CSV；任何不一致都会抛 AssertionError。"""
    CHECKS.clear()

    # ---- E1_moments.csv
    ref = read_csv_dicts(os.path.join(OUT, "E1_moments.csv"))
    got = arr["E1__mom"]
    keys = meta["E1__mom_keys"]
    labels = meta["E1__mom_labels"]
    assert len(ref) == got.shape[0], f"E1_moments.csv 行数不符: {len(ref)} vs {got.shape[0]}"
    for i, row in enumerate(ref):
        assert [row["profile"], row["scheme"]] == labels[i], (row, labels[i])
        for j, k in enumerate(keys):
            if k in row:
                _chk(f"E1_moments[{row['profile']}/{row['scheme']}].{k}", got[i, j], row[k])

    # ---- E1_stieltjes.csv
    ref = read_csv_dicts(os.path.join(OUT, "E1_stieltjes.csv"))
    got = arr["E1__sth"]
    for i, row in enumerate(ref):
        assert [row["profile"], row["scheme"]] == meta["E1__sth_labels"][i]
        for j, k in enumerate(meta["E1__sth_keys"]):
            _chk(f"E1_stieltjes[{row['profile']}/{row['scheme']}].{k}", got[i, j], row[k],
                 atol=1e-12, rtol=1e-8)

    # ---- E3_four_quantities.csv
    ref = read_csv_dicts(os.path.join(OUT, "E3_four_quantities.csv"))
    got = arr["E3__rows"]
    keys = meta["E3__keys"]
    for i, row in enumerate(ref):
        assert row["profile"] == meta["E3__labels"][i]
        for j, k in enumerate(keys):
            if k in row:
                _chk(f"E3[{row['profile']}].{k}", got[i, j], row[k])

    # ---- E4_counts.csv
    ref = read_csv_dicts(os.path.join(OUT, "E4_counts.csv"))
    got = arr["E4__rows"]
    keys = meta["E4__keys"]
    for i, row in enumerate(ref):
        assert [row["cons"], int(float(row["N"]))] == [meta["E4__labels"][i][0],
                                                      int(meta["E4__labels"][i][1])]
        for j, k in enumerate(keys):
            _chk(f"E4_counts[{row['cons']}/N={row['N']}].{k}", got[i, j], row[k])

    # ---- A1_scale.csv
    ref = read_csv_dicts(os.path.join(OUT, "A1_scale.csv"))
    got = arr["A1__rows"]
    keys = meta["A1__keys"]
    for i, row in enumerate(ref):
        for j, k in enumerate(keys):
            if k in row:
                _chk(f"A1_scale[{row['N']}/{row['profile']}].{k}", got[i, j], row[k])

    # ---- A2_nongaussian.csv
    ref = read_csv_dicts(os.path.join(OUT, "A2_nongaussian.csv"))
    got = arr["A2__rows"]
    keys = meta["A2__keys"]
    for i, row in enumerate(ref):
        assert row["dist"] == meta["A2__labels"][i]
        for j, k in enumerate(keys):
            _chk(f"A2[{row['dist']}].{k}", got[i, j], row[k])

    # ---- A3_population.csv（无表头：key, value[, "th", value]）
    a3 = {}
    with open(os.path.join(OUT, "A3_population.csv"), newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                a3[row[0]] = row
    sc = arr["A3__scalars"]
    mapping = [("m1_post_emp", sc[0]), ("m2_pre_emp", sc[1]), ("m2_post_emp", sc[2]),
               ("mu0_emp", sc[3]), ("control_max_deviation", sc[4])]
    for key, val in mapping:
        if key in a3:
            _chk(f"A3_population.{key}", val, a3[key][1])

    bad = [c for c in CHECKS if not c["ok"]]
    max_dev = max((c["abs_dev"] for c in CHECKS), default=0.0)
    lines = [
        "重放数据 vs 已存盘 CSV —— 逐项校验报告",
        "=" * 78,
        f"SEED = {SEED}（与各原脚本一致）",
        f"校验条目数 : {len(CHECKS)}",
        f"全部通过   : {'YES' if not bad else 'NO'}",
        f"最大绝对偏差: {max_dev:.3e}",
        "",
        "说明：E1/E3/E4_spectral/A3 的绘图数据原脚本未落盘。本脚本以相同 SEED 与相同",
        "随机数消费顺序重放复现；下列统计量与已存盘 CSV 完全一致，即证明随机数流与原",
        "运行逐位相同，因而用于绘图的原始数组也与原图完全相同（数据未被更改）。",
        "",
        f"{'item':<52}{'replayed':>16}{'on-disk':>16}{'|dev|':>11}",
        "-" * 95,
    ]
    for c in CHECKS:
        flag = "" if c["ok"] else "   <<< MISMATCH"
        lines.append(f"{c['item'][:51]:<52}{c['replayed']:>16.10g}{c['on_disk']:>16.10g}"
                     f"{c['abs_dev']:>11.2e}{flag}")
    with open(VERIFY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if bad:
        for c in bad[:20]:
            print(f"  MISMATCH {c['item']}: replayed={c['replayed']!r} on-disk={c['on_disk']!r}")
        raise AssertionError(f"{len(bad)} 项与 CSV 不一致，重放不可信 —— 已中止，未输出任何图。")
    print(f"  校验通过：{len(CHECKS)} 项全部与 CSV 一致，最大绝对偏差 {max_dev:.3e}")


def load_or_replay(recompute: bool):
    os.makedirs(PANEL_DIR, exist_ok=True)
    if os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_META) and not recompute:
        print("复用重放缓存（如需强制重算请加 --recompute）")
        with np.load(CACHE_NPZ) as z:
            arr = {k: z[k] for k in z.files}
        with open(CACHE_META, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        print("重放随机数流（精确复现原脚本未落盘的绘图数据）……")
        arr, meta = {}, {}
        for tag, fn in [("E1", replay_E1), ("E3", replay_E3), ("E4", replay_E4), ("A3", replay_A3)]:
            print(f"  - {tag} …", flush=True)
            a, m = fn()
            arr.update(a)
            meta.update(m)
        np.savez(CACHE_NPZ, **arr)
        with open(CACHE_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    verify_against_csvs(arr, meta)
    return arr, meta


# =========================================================================
# 第三部分：逐 panel 绘图
# =========================================================================
def plot_E1(arr, only):
    xgrid = arr["E1__xgrid"]
    N, d, c, R = 1000, 800, 0.8, 20
    sub = "" if narrow() else f"  ($N={N}$, $d={d}$, $c={c}$, $n_{{\\rm rep}}={R}$)"
    for pname in ["uniform", "twopoint"]:
        # --- 计数律 PMF（multinomial / residual）
        for scheme in ["multinomial", "residual"]:
            name = f"E1_{pname}_count_{scheme}"
            if only and only not in name:
                continue
            n_pooled = arr[f"E1__{pname}__{scheme}__n"]
            vals, cnts = np.unique(n_pooled, return_counts=True)
            ks_th = arr[f"E1__{pname}__{scheme}__ks_th"]
            p_th = arr[f"E1__{pname}__{scheme}__p_th"]
            fig, ax = new_panel(2.45)
            ax.stem(vals, cnts / len(n_pooled), linefmt="C0-", markerfmt="C0o",
                    basefmt=" ", label="empirical")
            ax.stem(ks_th, p_th, linefmt="r--", markerfmt="rx", basefmt=" ", label="theory")
            ax.set_xlim(-0.5, 8.5)
            ax.set_xlabel("count $k$")
            ax.set_ylabel(lab("probability mass", "mass"))
            ax.set_title(f"Count PMF ({scheme}){sub}", fontsize=BASE_FS)
            compact_legend(ax, loc="upper right")
            save_panel(fig, name)

        # --- 谱密度（pre / multinomial / residual）
        # 同一剖面的三张并排比较，y 轴须统一（x 已固定）；不同剖面之间不强制统一，
        # 因为 twopoint 的峰值本就更高，跨剖面共轴会压扁 multinomial 面板。
        _yt = 0.0
        for s in ["pre", "multinomial", "residual"]:
            ee = arr[f"E1__{pname}__{s}__eigs"]
            pp = ee[ee > 1e-10]
            hh, _ = np.histogram(pp, bins=120, density=True)
            _yt = max(_yt, hh.max(), np.nanmax(arr[f"E1__{pname}__curve__{s}"]))
        E1_YLIM = (0.0, _yt * 1.14)

        for scheme in ["pre", "multinomial", "residual"]:
            name = f"E1_{pname}_spectrum_{scheme}"
            if only and only not in name:
                continue
            eigs = arr[f"E1__{pname}__{scheme}__eigs"]
            pos = eigs[eigs > 1e-10]
            mu0_emp = float(arr[f"E1__{pname}__{scheme}__mu0_emp_plot"][0])
            mu0_th = float(arr[f"E1__{pname}__{scheme}__mu0_th"][0])
            fig, ax = new_panel(2.6)
            ax.hist(pos, bins=120, density=True, alpha=0.55, color="C0",
                    edgecolor="none", label="empirical")
            ax.plot(xgrid, arr[f"E1__{pname}__curve__{scheme}"], "r-", lw=1.1,
                    label="$T_c$ prediction")
            ax.plot([0], [mu0_emp], "ks", ms=4.5,
                    label=lab(f"atom emp = {mu0_emp:.3f}", f"emp {mu0_emp:.3f}"))
            ax.plot([0], [mu0_th], "r^", ms=4.5,
                    label=lab(f"atom th = {mu0_th:.3f}", f"th {mu0_th:.3f}"))
            ax.set_xlim(-0.3, 6)
            ax.set_ylim(*E1_YLIM)
            ax.set_xlabel("eigenvalue")
            ax.set_ylabel("density")
            ax.set_title(f"Spectrum ({scheme}){sub}", fontsize=BASE_FS)
            compact_legend(ax, loc="upper right")
            save_panel(fig, name)


def plot_E2(only):
    p = 1.0 - np.exp(-1.0)
    atom = read_csv_dicts(os.path.join(OUT, "E2_atom.csv"))
    edge = read_csv_dicts(os.path.join(OUT, "E2_edge.csv"))
    cc = [float(r["c"]) for r in atom]

    name = "E2_partA_zero_mass"
    if not only or only in name:
        fig, ax = new_panel(2.5)
        ax.plot(cc, [float(r["q0_th"]) for r in atom], "r-", lw=1.2,
                label=lab(r"theory $\max(0,\,1-(1-e^{-1})/c)$", "theory"))
        ax.errorbar(cc, [float(r["q0_emp"]) for r in atom],
                    yerr=[2 * float(r["q0_se"]) for r in atom],
                    fmt="ko", ms=3.0, lw=0.8, capsize=1.8, elinewidth=0.7,
                    label=lab("empirical (count rank)", "empirical"))
        ax.axvline(p, color="gray", ls=":", lw=0.8)
        ax.text(p + 0.008, 0.30, r"$c^{*}=1-e^{-1}$", fontsize=MIN_FS)
        ax.set_xlabel("$c$")
        ax.set_ylabel(lab("zero mass $q_{0,N}$", "$q_{0,N}$"))
        ax.set_title(lab("Rank / zero-atom transition", "Zero-atom transition"),
                     fontsize=BASE_FS)
        compact_legend(ax, loc="upper left")
        save_panel(fig, name)

    name = "E2_partB_lower_edge"
    if not only or only in name:
        ce = [float(r["c"]) for r in edge]
        fig, ax = new_panel(2.6)
        ax.plot(ce, [float(r["bench"]) for r in edge], "g--", lw=1.2,
                label=lab(r"bound $|\sqrt{c}-\sqrt{p}|^{2}$", "bound"))
        ax.plot(ce, [float(r["edge_true"]) for r in edge], "r-", lw=1.2,
                label=lab("true edge (Prop. 13)", "true edge"))
        ax.errorbar(ce, [float(r["lmin_med"]) for r in edge],
                    yerr=[[float(r["lmin_med"]) - float(r["lmin_q1"]) for r in edge],
                          [float(r["lmin_q3"]) - float(r["lmin_med"]) for r in edge]],
                    fmt="ko", ms=3.0, lw=0.8, capsize=1.8, elinewidth=0.7,
                    label=lab(r"$\lambda_{\min}^{+}$ median, quartiles",
                              r"$\lambda_{\min}^{+}$"))
        ax.axvline(p, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("$c$")
        ax.set_ylabel(lab(r"lower edge / $\lambda_{\min}^{+}$", r"$\lambda_{\min}^{+}$"))
        ax.set_title(lab("Lower edge: three levels", "Lower edge"), fontsize=BASE_FS)
        compact_legend(ax, loc="upper left")
        save_panel(fig, name)


def plot_E3(arr, only):
    rows = arr["E3__rows"]
    keys = ["ess", "m2_emp", "m2_se", "m2_th", "nu0_emp", "nu0_se", "nu0_th",
            "mu0_emp", "mu0_se", "mu0_th", "m3_emp", "m3_se", "m3_th"]
    ix = {k: i for i, k in enumerate(keys)}
    ess = float(rows[0, ix["ess"]])

    name = "E3_cdf"
    if not only or only in name:
        fig, ax = new_panel(2.5)
        for i, (plab, color) in enumerate([("A", "C0"), ("B", "C3")]):
            e = np.sort(arr[f"E3__eigs__{plab}"])
            cdf = np.arange(1, len(e) + 1) / len(e)
            ax.plot(e, cdf, color=color, lw=1.0,
                    label=lab(f"empirical CDF [{plab}]", f"[{plab}]"))
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlim(-0.2, 6)
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("CDF")
        ax.set_title(lab(f"Same ESS $={ess:.0f}$, different spectra",
                         f"Same ESS $={ess:.0f}$"), fontsize=BASE_FS)
        compact_legend(ax, loc="lower right")
        save_panel(fig, name)

    name = "E3_zero_atom_bars"
    if not only or only in name:
        emp_mu = [float(rows[i, ix["mu0_emp"]]) for i in range(2)]
        th_mu = [float(rows[i, ix["mu0_th"]]) for i in range(2)]
        nu0_th = [float(rows[i, ix["nu0_th"]]) for i in range(2)]
        gap = emp_mu[0] - emp_mu[1]
        th_gap = th_mu[0] - th_mu[1]
        xpos = np.arange(2)
        fig, ax = new_panel(2.6)
        ax.bar(xpos - 0.15, emp_mu, width=0.3, color="C0", edgecolor="k",
               linewidth=0.5, label=lab(r"empirical $\mu(\{0\})$", "empirical"))
        ax.bar(xpos + 0.15, th_mu, width=0.3, color="C3", edgecolor="k",
               linewidth=0.5, label=lab(r"theory $\mu(\{0\})$", "theory"))
        # 柱顶数值标签：极窄宽度下两个五字符标签没有并排空间（1.66 in 时
        # "0.346" 与 "0.345" 会连成一串），此时省略——精确值在 Table III 里。
        if PANEL_W >= 2.0:
            for i, (e_, t_) in enumerate(zip(emp_mu, th_mu)):
                ax.text(i - 0.15, e_ + 0.006, f"{e_:.3f}", ha="center", fontsize=MIN_FS)
                ax.text(i + 0.15, t_ + 0.006, f"{t_:.3f}", ha="center", fontsize=MIN_FS)
        ax.set_xticks(xpos)
        ax.set_xticklabels(
            [n if narrow() else f"profile {n}\n(count $\\nu_0$: {nu0_th[i]:.4f})"
             for i, n in enumerate(["A", "B"])], fontsize=MIN_FS)
        ax.set_ylabel(lab("spectral zero atom", r"$\mu(\{0\})$"))
        ax.set_title(f"Gap $={gap:.3f}$ (th. {th_gap:.3f})", fontsize=BASE_FS)
        # 柱顶带数值标签，原右上角图例正好压住 profile B 的两个标签（0.285/0.286
        # 被完全遮掉）。上扩 y 轴在柱顶之上留出空带，图例两列横排置于其中。
        ncol_b = 2 if PANEL_W >= 3.0 else 1
        ax.set_ylim(0.0, max(max(emp_mu), max(th_mu)) * 1.55)
        compact_legend(ax, loc="upper center", ncol=ncol_b,
                       columnspacing=1.0, borderaxespad=0.25)
        save_panel(fig, name)


def plot_E4(arr, only):
    rows = read_csv_dicts(os.path.join(OUT, "E4_counts.csv"))   # 直接读 CSV，不重算
    markers = {"uniform": "o", "interior": "s", "endpoint": "^"}
    cons_list = ["uniform", "interior", "endpoint"]

    name = "E4_counts_dev2"
    if not only or only in name:
        fig, ax = new_panel(2.5)
        # uniform 构造的 dev2 恒为 0（权重精确均匀），在对数轴上无法表示。
        # 早先版本把它钳到 1e-18，导致 y 轴横跨 18 个数量级，另两条曲线被压成
        # 顶部一条窄带，1/N 斜率不可见。改为不画该曲线，在图例中声明其恒为零。
        for cons in cons_list:
            sub = [r for r in rows if r["cons"] == cons]
            Nl = [float(r["N"]) for r in sub]
            yv = [float(r["dev2"]) for r in sub]
            if max(yv) <= 0.0:
                ax.plot([], [], markers[cons] + "--", lw=0.9, ms=3.2,
                        label=f"{cons} ($\\equiv 0$)")
                continue
            ax.semilogx(Nl, yv, markers[cons] + "--", lw=0.9, ms=3.2, alpha=0.85,
                        label=f"{cons}")
        # 1/N 参考斜率：直接支撑 IX.E 中"以 1/N 速率趋零"的论断
        Nref = np.array([float(r["N"]) for r in rows if r["cons"] == "endpoint"])
        yref = np.array([float(r["dev2"]) for r in rows if r["cons"] == "endpoint"])
        ax.semilogx(Nref, yref[0] * Nref[0] / Nref, "k:", lw=0.8,
                    label=lab(r"reference $\propto 1/N$", r"$\propto 1/N$"))
        ax.set_yscale("log")
        if narrow():   # 窄图：稀疏对数刻度，避免 10^{-n} 标签堆叠加宽画布
            from matplotlib.ticker import LogLocator, NullFormatter
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
            ax.yaxis.set_minor_formatter(NullFormatter())
        # 曲线沿对角线自左上走到右下，右上角原图例正好压在 endpoint/参考线上。
        # 上扩 y 轴造出数据之上的空带（数据止于 3.9e-3），图例两列横排置于其中；
        # y 刻度只标到 1e-3，空带不带刻度。与 RnN 面板图例位置一致，便于并排。
        ncol_d = 2 if PANEL_W >= 2.0 else 1
        ax.set_ylim(2.5e-5, 8e-2 if ncol_d == 2 else 4e-1)
        ax.set_yticks([1e-4, 1e-3])
        ax.set_xlabel(lab("$N$ (log scale)", "$N$"))
        ax.set_ylabel(lab(r"$(1/N)\sum(\lambda_i-1)^{2}$",
                          r"$(1/N)\sum(\lambda_i{-}1)^{2}$"))
        ax.set_title(r"Weight deviation $\to 0$", fontsize=BASE_FS)
        compact_legend(ax, loc="upper center", ncol=ncol_d, columnspacing=1.0,
                       borderaxespad=0.25)
        save_panel(fig, name)

    name = "E4_counts_RnN"
    if not only or only in name:
        fig, ax = new_panel(2.5)
        for cons in cons_list:
            sub = [r for r in rows if r["cons"] == cons]
            Nl = [float(r["N"]) for r in sub]
            ax.semilogx(Nl, [float(r["RnN"]) for r in sub], markers[cons] + "-",
                        lw=0.9, ms=3.2, alpha=0.9, label=f"{cons}")
        for y in (0.0, 0.4, 1.0):
            ax.axhline(y, color="gray", lw=0.5, ls=":")
        # 三条水平线占满 y∈[0,1]，坐标区内没有空处可放图例（原图例压在 y=0.4 的
        # interior 线上）。上扩 y 轴造出 y>1 的专用空带，图例置于其中，与所有
        # 曲线完全分离；y 刻度仍只标到 1.0，空带不带刻度，读者不会误读数据范围。
        # 列数按宽度分档：列太多会横向溢出坐标区（1.66 in 下两列就已放不下）。
        # 空带高度随行数增加，保证图例始终落在 y>1 的空白处。
        ncol = 3 if PANEL_W >= 3.0 else (2 if PANEL_W >= 2.0 else 1)
        ax.set_ylim(-0.06, {3: 1.30, 2: 1.62, 1: 2.05}[ncol])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.text(0.985, 0.43, r"$\alpha=0.4$", fontsize=MIN_FS,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xlabel(lab("$N$ (log scale)", "$N$"))
        ax.set_ylabel(r"$R_N/N$")
        ax.set_title(lab(r"Residual randomization $R_N/N$",
                         r"Randomization $R_N/N$"), fontsize=BASE_FS)
        compact_legend(ax, loc="upper center", ncol=ncol, columnspacing=1.0,
                       borderaxespad=0.25)
        save_panel(fig, name)

    # ---- spectral panels（数据来自重放，已由 E4_counts.csv 校验随机数流）
    xgrid = arr["E4__xgrid"]
    N, d, c = [float(v) for v in arr["E4__params"]]
    mu0 = arr["E4__mu0"]
    fkey = {"pre": "E4__f_pre", "multinomial": "E4__f_mult", "residual": "E4__f_res"}
    mu0_ix = {"pre": 0, "multinomial": 1, "residual": 2}

    # 原多联图靠 sharex/sharey 保证三面板同轴；拆成独立子图后必须显式统一，
    # 否则并排比较时 y 上限 1.66 / 1.48 / 2.53、x 范围 3.6 / 6.1 / 6.7，会误导读者。
    _sch = ["pre", "multinomial", "residual"]
    _ytop = 0.0
    for s in _sch:
        e = arr[f"E4__spec__{s}"]
        pos = e[e > 1e-10]
        hh, _ = np.histogram(pos, bins=100, density=True)
        _ytop = max(_ytop, hh.max(), np.nanmax(arr[fkey[s]]),
                    float(np.mean(e <= 1e-10)), float(mu0[mu0_ix[s]]))
    E4_XLIM = (-0.25, 7.0)
    E4_YLIM = (0.0, _ytop * 1.14)

    for scheme in ["pre", "multinomial", "residual"]:
        name = f"E4_spectral_{scheme}"
        if only and only not in name:
            continue
        e = arr[f"E4__spec__{scheme}"]
        pos = e[e > 1e-10]
        nzero = float(np.mean(e <= 1e-10))
        m0 = float(mu0[mu0_ix[scheme]])
        fig, ax = new_panel(2.6)
        ax.hist(pos, bins=100, density=True, alpha=0.6, color="C0", edgecolor="none",
                label=lab("empirical (positive part)", "empirical"))
        ax.plot(xgrid, arr[fkey[scheme]], "r-", lw=1.1, label="$T_c$ prediction")
        ax.plot([0], [nzero], "ks", ms=4.5,
                label=lab(f"zero mass emp = {nzero:.3f}", f"emp {nzero:.3f}"))
        ax.plot([0], [m0], "r^", ms=4.5,
                label=lab(f"zero mass th = {m0:.3f}", f"th {m0:.3f}"))
        ax.set_xlim(*E4_XLIM)
        ax.set_ylim(*E4_YLIM)
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("density")
        ttl = "pre (MP)" if scheme == "pre" else ("residual, $\\alpha=0.5$" if scheme == "residual" else scheme)
        cfg = "" if narrow() else f"  ($N={int(N)}$, $c={c:.3f}$)"
        ax.set_title(f"Spectrum: {ttl}{cfg}", fontsize=BASE_FS)
        compact_legend(ax, loc="upper right")
        save_panel(fig, name)


def plot_A3(arr, only):
    name = "A3_population_spectrum"
    if only and only not in name:
        return
    e = arr["A3__eigs_pool"]
    pos = e[e > 1e-10]
    mu0_emp = float(arr["A3__scalars"][3])
    fig, ax = new_panel(2.6)
    ax.hist(pos, bins=120, density=True, alpha=0.55, color="C0", edgecolor="none",
            label=lab("empirical (positive part)", "empirical"))
    ax.plot(arr["A3__xgrid"], arr["A3__f_prod"], "r-", lw=1.1,
            label=lab(r"$H\boxtimes\mu_Q$ (S-transform)", r"$H\boxtimes\mu_Q$"))
    ax.plot([0], [mu0_emp], "ks", ms=4.5,
            label=lab(f"atom emp = {mu0_emp:.3f}", f"emp {mu0_emp:.3f}"))
    ax.plot([0], [0.20985], "r^", ms=4.5, label=lab("atom th = 0.210", "th 0.210"))
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    ax.set_title(lab("Two-point population spectrum", "Two-point population"),
                 fontsize=BASE_FS)
    compact_legend(ax, loc="upper right")
    save_panel(fig, name)


def main():
    global PANEL_W, SAVED

    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="忽略缓存，强制重新重放")
    ap.add_argument("--only", default=None, help="只输出名字包含该子串的 panel")
    ap.add_argument("--width", default="all",
                    help="出图宽度：预设名（" + ", ".join(WIDTH_PRESETS)
                         + "）、英寸数值、逗号分隔的多个，或 all（默认，全部预设）")
    args = ap.parse_args()

    if args.width == "all":
        widths = [(k, v) for k, v in WIDTH_PRESETS.items()]
    else:
        widths = []
        for tok in args.width.split(","):
            tok = tok.strip()
            if tok in WIDTH_PRESETS:
                widths.append((tok, WIDTH_PRESETS[tok]))
            else:
                widths.append((f"{float(tok):.2f}in", float(tok)))

    os.makedirs(PANEL_DIR, exist_ok=True)
    apply_ieee_style()
    print(f"字体族：{FONT_FAMILY}（数学符号 mathtext/STIX）；基础 {BASE_FS} pt，最小 {MIN_FS} pt")
    print("出图宽度：" + ", ".join(f"{k} = {v} in" for k, v in widths))
    print("注意：子图按最终印出宽度绘制，插入 LaTeX 时请勿再用 width= 缩放。")

    arr, meta = load_or_replay(args.recompute)

    all_saved = []
    for tag, w in widths:
        PANEL_W = w
        SAVED = []
        print(f"\n绘制 {tag}（{w} in）……")
        plot_E1(arr, args.only)
        plot_E2(args.only)
        plot_E3(arr, args.only)
        plot_E4(arr, args.only)
        plot_A3(arr, args.only)
        over = [s for s in SAVED if s[1] > w + 0.02]
        print(f"  {len(SAVED)} 个 PDF → {panel_dir()}")
        print(f"  宽度校验：{len(SAVED) - len(over)}/{len(SAVED)} 个 <= {w} in"
              + (f"；超宽 {len(over)} 个：" + ", ".join(s[0] for s in over) if over
                 else "（全部达标）"))
        all_saved.append((tag, w, list(SAVED), over))

    print("\n===== 汇总 =====")
    for tag, w, saved, over in all_saved:
        hs = [s[2] for s in saved]
        print(f"  {tag:<8} {w:>5.2f} in × {len(saved):>2} 个"
              f"  高度 {min(hs):.2f}–{max(hs):.2f} in"
              f"  {'OK' if not over else f'{len(over)} 超宽'}")
    print(f"\n校验报告：{VERIFY_TXT}")
    print(f"排版说明：{os.path.join(PANEL_DIR, 'PANELS_README.md')}")


if __name__ == "__main__":
    main()
