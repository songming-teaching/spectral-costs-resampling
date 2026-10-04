# 单子图（panel）输出说明

由 `plot_panels.py` 生成。每个宽度一个子目录，同名文件互不覆盖。

## 核心约束：不要在 LaTeX 里缩放

子图已按**最终印出宽度**绘制，字号（正文 9 pt / 刻度 8 pt）是按该宽度定的。
插入时用**不带 `width=`** 的 `\includegraphics`：

```latex
\includegraphics{fig/E2_partA_zero_mass}        % 正确：1:1 插入
\includegraphics[width=\linewidth]{...}          % 错误：会再缩一次，字变小
```

这正是原多联图偏小的原因——图在 LaTeX 里被压到 `0.49\linewidth`，字号跟着一起缩。
若某张图尺寸不合，改 `--width` 重新出图，而不是在 LaTeX 里缩放。

## 宽度预设

IEEE TSP 版面：单栏 3.4 in，跨栏 7.16 in。

| 目录 | 宽度 | 落位 | LaTeX |
|---|---|---|---|
| `w3.40_single/` | 3.40 in | 单栏，一图独占 | `figure` + `\includegraphics{...}` |
| `w3.50_span2/` | 3.50 in | 跨栏，两图并排 | `figure*` + 两个 `\includegraphics{...}` |
| `w2.33_span3/` | 2.33 in | 跨栏，三图并排 | `figure*` + 三个 `\includegraphics{...}` |
| `w1.66_col2/` | 1.66 in | 单栏内两图并排 | 见下方注意事项 |

`--width` 也接受任意英寸数值，例如 `--width 2.6`。

**`col2`（1.66 in）不推荐。** 该宽度下图例占去可观面积，谱图与边缘图的细节（尤其 E2 Part B 的临界区）基本不可读。除非版面极度紧张，优先用 `span3`。

## 三种典型排法

单栏一图：

```latex
\begin{figure}[!t]
  \centering
  \includegraphics{fig/E3_zero_atom_bars}
  \caption{...}\label{fig:e3atom}
\end{figure}
```

跨栏两图并排（`span2`，7.16 = 3.50 + 3.50 + 间距）：

```latex
\begin{figure*}[!t]
  \centering
  \subfloat[Part A: zero atom]{\includegraphics{fig/E2_partA_zero_mass}%
    \label{fig:e2a}}
  \hfil
  \subfloat[Part B: lower edge]{\includegraphics{fig/E2_partB_lower_edge}%
    \label{fig:e2b}}
  \caption{...}\label{fig:e2}
\end{figure*}
```

跨栏三图并排（`span3`）：

```latex
\begin{figure*}[!t]
  \centering
  \subfloat[pre]{\includegraphics{fig/E4_spectral_pre}}
  \hfil
  \subfloat[multinomial]{\includegraphics{fig/E4_spectral_multinomial}}
  \hfil
  \subfloat[residual]{\includegraphics{fig/E4_spectral_residual}}
  \caption{...}\label{fig:e4spec}
\end{figure*}
```

## 20 个子图清单

| 文件名 | 内容 | 对应重写版 IX 节 |
|---|---|---|
| `E1_{uniform,twopoint}_count_{multinomial,residual}` | 计数律 PMF vs 理论 | Fig. 1（上排） |
| `E1_{uniform,twopoint}_spectrum_{pre,multinomial,residual}` | 谱密度 vs $\mathcal T_c$ 预测 | Fig. 1（下排） |
| `E2_partA_zero_mass` | 零原子相变 | Fig. 2(a) |
| `E2_partB_lower_edge` | 下边缘三层次 | Fig. 2(b) |
| `E3_cdf` | 同 ESS 两剖面 CDF 对照 | Fig. 3(a) |
| `E3_zero_atom_bars` | 零原子柱图（gap 0.061） | Fig. 3(b) |
| `E4_counts_dev2` | 权重偏差 $\to0$，含 $1/N$ 参考线 | Fig. 4(a) |
| `E4_counts_RnN` | $R_N/N\to\{0,0.4,1\}$ | Fig. 4(b) |
| `E4_spectral_{pre,multinomial,residual}` | $\alpha=0.5$ 谱三联 | Fig. 5 |
| `A3_population_spectrum` | 两点总体谱 $H\boxtimes\mu_Q$ | 附录（可选） |

E1 共 10 张（2 剖面 × (2 计数 + 3 谱)）。

## 与原多联 PNG 的关系

原始多联 PNG 仍保留在 `output/`，未被修改。两者数据同源：E2、E4 计数面板直接读已存盘 CSV；E1、E3、E4 谱面板、A3 的绘图数组原脚本未落盘，本脚本以相同 `SEED=20260915` 重放复现，并将 393 项统计量与已存盘 CSV 逐项比对，最大绝对偏差 4.441e-16（见 `REPLAY_VERIFICATION.txt`）。因此子图与原图画的是同一批数。

## 一个已修正的作图 bug（影响原 E4_spectral.png）

`stage1_E4.py:157` 调用
`density_grid(xgrid, 1e-3, c, ks_poi*0+1.0, np.array([1.0]))`
时 supp 传入 61 个 `1.0` 而 probs 只有 1 个，长度不匹配，返回密度**恒为 0**。
后果：原 `E4_spectral.png` 里 pre 面板的 $\mathcal T_c$ 理论曲线是贴着横轴的平线，
不是 $\mathrm{MP}(c)$。正确调用 supp 与 probs 均为 `[1.0]`，峰值 1.69，与 `mp_density`
闭式解一致（本目录子图已修正）。

**`stage1_E4.py` 本身未改动**，所以 `output/E4_spectral.png` 仍带该错误。若正文改用
本目录的单子图，问题自动消失；若仍要用原多联 PNG，需先修 `stage1_E4.py:157` 再重跑。
修正后 393 项校验仍全部通过（该行只影响理论曲线，不消耗随机数）。

## 三处与原图不同的改动

1. **`E4_counts_dev2` 不再画 uniform 曲线。** 该构造的 $(1/N)\sum(\lambda_i-1)^2$ 恒为 0，在对数轴上无法表示；原图把它钳到 $10^{-18}$，使 y 轴横跨 18 个数量级，另两条曲线被压成顶部窄带、$1/N$ 斜率不可见。现改为在图例中标注 `uniform (≡0)`，并加一条 $\propto 1/N$ 参考线，直接支撑 IX.E 中"以 $1/N$ 速率趋零"的论断。正文需相应说明 uniform 构造的偏差恒为零（重写版 IX.E 已含此句）。
2. **重复次数记号改为 $n_{\rm rep}$**，避免与理论部分的 $R_N=\sum_iu_i$ 冲突（原图标题用 `R=20`）。

3. **并排子图显式共轴。** 原多联图靠 `sharex/sharey` 保证同轴，拆成独立 PDF 后
   各自 autoscale 会失去这个保证——E4 谱三联的 y 上限本来会变成 1.66 / 1.48 / 2.53、
   x 范围 3.6 / 6.1 / 6.7，并排比较时严重误导。现在同组子图强制统一轴范围：
   - `E4_spectral_*`：三张共用 x ∈ [−0.25, 7.0]、y ∈ [0, 2.88]
   - `E1_*_spectrum_*`：同一剖面的三张共用 y（x 已固定 [−0.3, 6]）；
     不同剖面之间不强制统一，因为 twopoint 峰值本就更高，跨剖面共轴会压扁 multinomial 面板。

   顺带把理论曲线的求值网格从 `0.02..4.0` 扩到 `0.02..7.0`：经验谱延伸到约 6.7
   （Prop. 13 的右支撑无界），统一 x 轴后红线在 4 处断掉会被误读为理论失配。
   纯显示网格，不消耗随机数。

窄图（< 2.80 in）自动改用短标签与短标题，**字号不变**——缩文字而不缩字号。
