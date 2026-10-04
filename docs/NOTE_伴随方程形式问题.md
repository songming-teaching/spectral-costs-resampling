# 关于伴随侧 Stieltjes 方程形式的一个问题的说明

**日期**：2026-09-15
**背景**：谱代价论文（v0.9）第一阶段数值实验的求解器自检（stage0）过程中发现。
**结论先行**：在求解论文 Definition 1 的伴随侧（companion）Stieltjes 变换时，文献中常见的伴随不动点方程形式 $z=-\frac1e+\frac1c\int\frac{t}{1+te}\,\eta(dt)$ 在本文参数化下**不成立**；变体 $z=-\frac1e+c\int\frac{t}{1+te}\,\eta(dt)$ 在 $\eta=\delta_1$（Marchenko–Pastur 特例）下侥幸精确成立，但对一般 $\eta$ 同样不成立（残差 $\sim10^{-2}$）。正确做法是论文 Definition 1 的**精确代数改写**（见下），已经数值模拟直接验证。若论文补充材料（S.21 附近）或其他地方引用了伴随方程的文献形式，建议顺手核对。

---

## 1. 问题的精确陈述

模型（论文记号）：$S_{r,N}=\frac1N Z D Z^\top$（$Z$ 为 $d\times N$，$D$ 为 $N\times N$ 计数对角阵，$d/N\to c$）。
- 协方差侧 LSD（$d\times d$）的 Stieltjes 变换 $m(z)$ 满足（论文 Definition 1，本文称为 (C)）：
$$z=-\frac1m+\int\frac{t}{1+ctm}\,\eta(dt),\qquad m:\mathbb C^+\to\mathbb C^+.$$
- 伴随侧 LSD（$N\times N$ 矩阵 $\frac1N Z^\top DZ$）的 Stieltjes 变换记 $e(z)$。由于两矩阵非零谱相同，测度层面有精确关系 $\mu_{\rm comp}=c\,\mu+(1-c)\delta_0$（$c<1$），因而
$$e(z)=c\,m(z)+\frac{c-1}{z}.\tag{R}$$

**需要求解器回答的问题**：给定 $\eta$（计数律，本文中为混合 Poisson 型），如何在 $\mathbb C^+$ 上逐点计算 $e(z)$？

## 2. 两个常见的文献形式为何不适用

**形式 K1**：$z=-\dfrac1e+\dfrac1c\displaystyle\int\dfrac{t}{1+te}\,\eta(dt)$。
- $\eta=\delta_1$、$c=0.8$、$z=\mathrm i$ 处残差 $3.1\times10^{-1}$；多点复验同量级。
- 诊断：该形式对应的 MP 伴随二次方程为 $ze^2+\big(z-\frac{1-c}{c}\big)e+1=0$，而**真实**伴随变换（由 (R) 从 MP 闭式解推出）满足 $ze^2+(z+1-c)e+1=0$。两个二次方程只在 $c\mapsto1/c$ 的互换下相容——即 K1 属于比值参数取**倒数**的约定（$N/d$ 而非 $d/N$），直接照搬到本文参数化即错。

**形式 K2**：$z=-\dfrac1e+c\displaystyle\int\dfrac{t}{1+te}\,\eta(dt)$。
- $\eta=\delta_1$ 时残差 $5\times10^{-15}$（**精确**成立——MP 的退化性使多种方程形式在该点重合）；
- 两点律 $\eta=\frac12\delta_{1/2}+\frac12\delta_{3/2}$ 时残差 $6\times10^{-3}\sim3.6\times10^{-2}$（四个测试点）——**不成立**。

**裁判实验**（直接数值模拟，不依赖任何待证方程）：
- $d=2000,N=2500$，$D$ 取两点对角阵，$Z$ 标准高斯。经验协方差侧变换 $\hat m(1\mathrm i)=0.31052+0.62907\mathrm i$，Definition 1 解出的 $m(1\mathrm i)=0.31068+0.62899\mathrm i$（差 $3\times10^{-4}$，属有限样本涨落）——**(C) 成立**；
- $d=800,N=1000$ 的伴随矩阵经验变换 $\hat e(1\mathrm i)=0.24826+0.70329\mathrm i$，由 (R) 从 $m$ 推出的 $e(1\mathrm i)=0.24855+0.70319\mathrm i$——**关系 (R) 成立**；
- 将经验 $e$ 代入 K2：残差 $9.7\times10^{-3}$——**K2 不成立**（排除求解器误差因素）。

## 3. 正确的伴随侧求解路线（本文采用，已验证）

不引入任何文献形式，只做 (C) 的精确代数改写。令辅助量 $u:=c\,m$。则 (C) 等价于
$$z=-\frac{c}{u}+\int\frac{t}{1+tu}\,\eta(dt),\tag{C'}$$
这是关于 $u$ 的不动点方程 $u=-c\big/\big(z-\int\frac{t}{1+tu}\eta(dt)\big)$，与 (C) 同构、同一迭代框架可解。解出 $u$ 后由 (R) 读出伴随变换：
$$e=u+\frac{c-1}{z}.$$
**验证**：(C') 的方程残差达 $1.1\times10^{-16}$（机器精度级）；由它得到的 $e$ 与独立实现的协方差侧 $m$ 满足关系 (R) 到 $6\times10^{-13}$（相对）。

**附赠恒等式（作求解器内检）**：由 (C) 与 (R) 可严格推出
$$\int\frac{\eta(dt)}{1+t\,u(z)}=-z\,e(z),\qquad u=cm,$$
在 stage0 中作为独立检查通过（最大残差 $1.9\times10^{-13}$）。

**推导备注**：若坚持要一个只含 $e$ 的隐式方程，由 (C) 与 (R) 消去 $m$ 得到
$$z=-\frac{cz}{ze+1-c}+\int\frac{t}{1+t\big(e-\frac{c-1}{z}\big)}\,\eta(dt),$$
形式繁琐且不动点性质差，无实用价值；(C')+(R) 是干净正确的路线。

## 4. 教训与对论文的影响

1. **MP 特例 ($\eta=\delta_1$) 对求解器验证是必要的但绝不充分**：在该点多个互不相容的方程形式退化重合，MP 恢复测试全部通过并不能证明方程形式正确。本文计数律（混合 Poisson，支集为全体非负整数）提供了天然的第二测试对象。
2. 论文 Definition 1 的**测度恒等式**（$c\,\mathcal T_c(\eta)+(1-c)\delta_0=D_c(\eta\boxtimes\mathrm{MP}(1/c))$）与不动点方程 (C) 均不受影响——本问题只涉及"伴随侧用哪个不动点方程求解器"这一计算层面。若正文/补充材料在任何地方以引理形式写下了伴随不动点方程的文献形式（K1/K2），需要核对并按 (C')+(R) 修正或改写为引用。
3. 文献形式出错的根源大概率是比值参数的约定（$n/N$ 与 $N/n$ 之差）以及"该方程描述的是哪个变换"的混淆；引用时须逐字母核对约定。

## 5. 可复现性

- 求解器与全部检查：`C:\phd\PFthinker\paper\experiments\stage0_solvers.py`（运行约 3 秒）；
- 检查日志与结果：`C:\phd\PFthinker\paper\experiments\output\stage0_log.txt`、`stage0_checks.json`；
- 裁判实验脚本见 stage0 开发过程记录（本说明第 2 节数值即由其产生）。
