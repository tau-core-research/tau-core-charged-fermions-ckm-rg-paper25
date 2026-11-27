#!/usr/bin/env python3
"""
Generate 4 Publication-Grade Vector PDF Figures for Tau Core Paper 25 (v1.1.0 — Referee-Hardened):
  Fig 1: K4 / End(V_3) Frobenius Equipartition Potential V(c0, r) & Brannen Circle (theta = 2/9)
  Fig 2: 3-Regime RG Evolution (M_Z -> M_PS [2-loop SM] -> M_R [G_PS-invariant Bidoublet])
         showing (a) y_b / y_tau freezing at 0.694 above M_PS vs SM desert failure (0.617),
         and (b) |V_cb(mu)| sign-flip (+3/2 y_t^2 below M_PS vs -y_t^2 above M_PS)
  Fig 3: Top-Down Predictions vs PDG 2024 MS-bar at M_Z (Pull Chart with strict threshold discipline)
  Fig 4: Structural No-Go Shields: (a) Genuine Real 10_H (|r_1|=1) Global Exclusion,
         (b) Fritzsch NNI (1,3)=0 Shortfall vs Observed |V_ub|/|V_cb|
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(REPO_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, SCRIPT_DIR)
import verify_paper25_fermion_masses_ckm_rg as v

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})

# =============================================================================
# FIGURE 1: K4 FROBENIUS EQUIPARTITION POTENTIAL & BRANNEN ANGLE SENSITIVITY
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))

r_grid = np.linspace(0.0, 1.25, 400)
for lam2_val, col, ls in [(0.5, "#377eb8", "--"), (1.5, "#1b4f72", "-"), (4.0, "#e41a1c", "-.")]:
    V_prof = -(1.0 + 2.0*r_grid**2)**2 / (4.0 * (1.0*(1.0 + 2.0*r_grid**2)**2 + lam2_val*(2.0*r_grid**2 - 1.0)**2))
    ax1.plot(r_grid, V_prof - (-0.25), color=col, lw=2.0, ls=ls, label=rf"$\lambda_2/\lambda_1 = {lam2_val}$")

ax1.axvline(1.0/np.sqrt(2.0), color="#2ca02c", lw=1.6, ls=":", label=r"$r = 1/\sqrt{2}\;(Q = 2/3)$")
ax1.set_xlabel(r"Traceless-to-trace amplitude $r = \|\tilde{S}_\ell\|_F / (\sqrt{2}\,\|S_{\ell,0}\|_F)$")
ax1.set_ylabel(r"Profiled flavon potential $\Delta V_{\min}(r) / (\mu^4 / \lambda_1)$")
ax1.set_title(r"(a) Exact Frobenius Equipartition Minimum ($Q = 2/3$)")
ax1.set_xlim(0.0, 1.25)
ax1.grid(True, alpha=0.25)
ax1.legend(loc="upper right", frameon=True)

th_grid = np.linspace(0.18, 0.26, 500)
err_mue, err_taumu = [], []
for th in th_grid:
    sk = np.sort([1.0 + np.sqrt(2.0)*np.cos(th + 2.0*np.pi*k/3.0) for k in range(3)])
    mk = sk**2
    err_mue.append(abs((mk[1]/mk[0]) / v.r_mu_e_exp - 1.0) * 100.0)
    err_taumu.append(abs((mk[2]/mk[1]) / v.r_tau_mu_exp - 1.0) * 100.0)

ax2.semilogy(th_grid, err_mue, color="#1f77b4", lw=2.0, label=r"$|\Delta(m_\mu/m_e)|$ vs. PDG pole [%]")
ax2.semilogy(th_grid, err_taumu, color="#d62728", lw=2.0, ls="--", label=r"$|\Delta(m_\tau/m_\mu)|$ vs. PDG pole [%]")
ax2.axvline(2.0/9.0, color="#2ca02c", lw=1.8, ls=":", label=r"$\theta = 2/d_{\mathrm{full}} = 2/9 = 0.22222$")
ax2.set_xlabel(r"Circulant phase $\theta\;[\mathrm{rad}]$")
ax2.set_ylabel(r"Relative error vs. PDG 2024 pole masses [%]")
ax2.set_title(r"(b) Precision Pinning at $\theta = 2/9$ ($0.001\%$ and $0.006\%$)")
ax2.set_xlim(0.18, 0.26)
ax2.set_ylim(3e-4, 50.0)
ax2.grid(True, which="both", alpha=0.25)
ax2.legend(loc="lower left", frameon=True)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_koide_equipartition_potential.pdf"))
plt.close(fig)

# =============================================================================
# FIGURE 2: 3-REGIME RG RUNNING (G_PS-INVARIANT BIDOUBLET vs PURE 2L DESERT)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))

t_grid_des = np.linspace(0.0, v.t_MS, 280)
sol_d = solve_ivp(v.rge_sm_2loop, [0.0, v.t_MS], v.y0, t_eval=t_grid_des, method="DOP853", rtol=1e-10, atol=1e-12)
log10_mu_des = np.log10(v.M_Z * np.exp(t_grid_des))
yb_ytau_des, vcb_des = [], []
for i in range(len(t_grid_des)):
    oi = v.extract_sm(sol_d.y[:, i])
    yb_ytau_des.append(oi['yd'][2] / oi['ye'][2])
    vcb_des.append(oi['Vcb'])

t_grid_sm1 = np.linspace(0.0, v.t_PS, 160)
sol_s1 = solve_ivp(v.rge_sm_2loop, [0.0, v.t_PS], v.y0, t_eval=t_grid_sm1, method="DOP853", rtol=1e-10, atol=1e-12)
o_ps_match = v.extract_sm(sol_s1.y[:, -1])
Y1_m  = 0.25 * (3.0 * o_ps_match['Yd'] + o_ps_match['Ye'])
Y15_m = 0.25 * (o_ps_match['Yd'] - o_ps_match['Ye'])

t_grid_ps = np.linspace(v.t_PS, v.t_MR, 100)
yb_ytau_ps, vcb_ps = [], []
for tp in t_grid_ps:
    if tp == v.t_PS:
        yb_ytau_ps.append(o_ps_match['yd'][2] / o_ps_match['ye'][2])
        vcb_ps.append(o_ps_match['Vcb'])
    else:
        op = v.run_ps_bidoublet(v.t_PS, tp, (v.g4_PS, v.g2L_PS, v.g2R_PS), o_ps_match['Yu'], Y1_m, Y15_m, v.b_PS_full)
        yb_ytau_ps.append(op['yd'][2] / op['ye'][2])
        vcb_ps.append(op['Vcb'])

log10_mu_ps = np.log10(v.M_Z * np.exp(t_grid_ps))

ax1.plot(log10_mu_des, yb_ytau_des, color="#888888", lw=1.8, ls="--", label="Pure 2-loop SM desert ($M_R$: 0.617)")
ax1.plot(np.log10(v.M_Z * np.exp(t_grid_sm1)),
         [v.extract_sm(sol_s1.y[:, i])['yd'][2]/v.extract_sm(sol_s1.y[:, i])['ye'][2] for i in range(len(t_grid_sm1))],
         color="#1f77b4", lw=2.2, label=r"3-Regime: 2-loop SM ($<M_{\mathrm{PS}}$)")
ax1.plot(log10_mu_ps, yb_ytau_ps, color="#d62728", lw=2.4, label=r"$G_{\mathrm{PS}}$ bidoublet RGE ($M_{\mathrm{PS}} \to M_R$: 0.694)")
ax1.axhline(2.0/3.0, color="#2ca02c", lw=1.6, ls=":", label=r"$\mathrm{Spin}(10)$ target $y_b/y_\tau = 2/3 = 0.667$")
ax1.axhline(1.0, color="black", lw=1.0, ls="-.", alpha=0.5, label=r"Minimal GUT $y_b = y_\tau = 1$ (excluded)")
ax1.axvline(np.log10(v.M_PS), color="#555555", lw=1.0, ls=":")
ax1.axvline(np.log10(v.M_R), color="#555555", lw=1.0, ls=":")
ax1.text(np.log10(v.M_PS)-0.35, 1.48, r"$M_{\mathrm{PS}}$", fontsize=8.5)
ax1.text(np.log10(v.M_R)-0.25, 1.48, r"$M_R$", fontsize=8.5)
ax1.set_xlabel(r"Renormalization scale $\log_{10}(\mu / \mathrm{GeV})$")
ax1.set_ylabel(r"Third-generation ratio $y_b(\mu) / y_\tau(\mu)$")
ax1.set_title(r"(a) $G_{\mathrm{PS}}$-Invariant Stabilization of $y_b / y_\tau$")
ax1.set_xlim(2.0, 18.2)
ax1.set_ylim(0.50, 1.65)
ax1.grid(True, alpha=0.25)
ax1.legend(loc="upper right", fontsize=7.6, frameon=True)

ax2.plot(log10_mu_des, vcb_des, color="#1f77b4", lw=2.0, label=r"2-loop SM $|V_{cb}(\mu)|$ ($+\frac{3}{2}y_t^2$ slope)")
ax2.plot(log10_mu_ps, vcb_ps, color="#d62728", lw=2.2, ls="--", label=r"$G_{\mathrm{PS}}$ bidoublet $|V_{cb}(\mu)|$ ($-y_t^2$ slope)")
ax2.axhline(v.lam_q**2, color="#2ca02c", lw=1.8, ls=":", label=r"Pure UV boundary $\lambda_q^2 = (2/9)^2 = 0.04938$")
ax2.axvline(np.log10(v.M_PS), color="#555555", lw=1.0, ls=":")
ax2.axvline(np.log10(v.M_R), color="#555555", lw=1.0, ls=":")
ax2.set_xlabel(r"Renormalization scale $\log_{10}(\mu / \mathrm{GeV})$")
ax2.set_ylabel(r"CKM mixing element $|V_{cb}(\mu)|$")
ax2.set_title(r"(b) Top-Yukawa $|V_{cb}(\mu)|$ Running Across $M_{\mathrm{PS}}$")
ax2.set_xlim(2.0, 18.2)
ax2.set_ylim(0.040, 0.051)
ax2.grid(True, alpha=0.25)
ax2.legend(loc="lower right", fontsize=7.8, frameon=True)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_three_regime_rg_running.pdf"))
plt.close(fig)

# =============================================================================
# FIGURE 3: PULL COMPARISON AT M_Z (PURE 2L DESERT vs 3-REGIME G_PS + 2L SM)
# =============================================================================
ye_d_uv = float(v.o_MR_des['ye'][2]) * (v.s_brannen / v.s_brannen[2])**2
yd_d_uv = np.array([(v.lam_q**3)*ye_d_uv[1], v.lam_q*ye_d_uv[1], (2.0/3.0)*float(v.o_MR_des['ye'][2])])
yu_d_uv = np.array([(v.lam_q**8)*float(v.o_MR_des['yu'][2]), (4.0/3.0)*(v.lam_q**4)*float(v.o_MR_des['yu'][2]), float(v.o_MR_des['yu'][2])])
y_des_uv = v.pack_sm(v.o_MR_des['g'][0], v.o_MR_des['g'][1], v.o_MR_des['g'][2], v.o_MR_des['lamH'],
                     np.diag(yu_d_uv).astype(complex), v.V_uv @ np.diag(yd_d_uv).astype(complex) @ v.V_uv.T, np.diag(ye_d_uv).astype(complex))
sol_des_dn = solve_ivp(v.rge_sm_2loop, [v.t_MR, 0.0], y_des_uv, method="DOP853", rtol=1e-10, atol=1e-12)
pred_des_MZ = v.extract_sm(sol_des_dn.y[:, -1])
des_vals = [
    pred_des_MZ['mu'][0]*1e3, pred_des_MZ['mu'][1], pred_des_MZ['md'][0]*1e3,
    pred_des_MZ['md'][1]*1e3, pred_des_MZ['md'][2], pred_des_MZ['Vus'],
    pred_des_MZ['Vcb'], pred_des_MZ['Vub'], pred_des_MZ['delta_deg'], pred_des_MZ['J']*1e5
]
pulls_des = [(dv - r['exp'])/r['sigma_exp'] for dv, r in zip(des_vals, v.table_results)]
pulls_ps_exp = [r['pull_exp'] for r in v.table_results]
pulls_ps_tot = [r['pull_tot'] for r in v.table_results]
labels = [r"$m_u$", r"$m_c$", r"$m_d$", r"$m_s$", r"$m_b$", r"$|V_{us}|$", r"$|V_{cb}|$", r"$|V_{ub}|$", r"$\delta_{\mathrm{CKM}}$", r"$J_{\mathrm{CKM}}$"]

fig, ax = plt.subplots(figsize=(9.6, 4.3))
y_pos = np.arange(len(labels))
h = 0.26
ax.axvspan(-2.0, 2.0, color="#2ca02c", alpha=0.10, label=r"$\pm 2\sigma$ consistency band")
ax.axvspan(-3.0, 3.0, color="#ff7f0e", alpha=0.06, label=r"$\pm 3\sigma$ band")
ax.axvline(0.0, color="black", lw=1.0)
ax.barh(y_pos + h, pulls_des, height=h, color="#d62728", alpha=0.80, label="Pure 2-loop SM desert ($m_b = +9.1\\sigma$ failure!)")
ax.barh(y_pos, pulls_ps_exp, height=h, color="#1f77b4", alpha=0.90, label="3-Regime $G_{\\mathrm{PS}}$ + 2L SM (raw PDG $\\sigma_{\\mathrm{exp}}$)")
ax.barh(y_pos - h, pulls_ps_tot, height=h, color="#17becf", alpha=0.90, label="3-Regime $G_{\\mathrm{PS}}$ + 2L SM ($\\sigma_{\\mathrm{tot}}$, 2% GUT thresh.)")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel(r"Pull $(O_{\mathrm{pred}} - O_{\mathrm{PDG}}) / \sigma$ at $\mu = M_Z$")
ax.set_title(r"Top-Down Closure at $M_Z$: Pure SM Desert vs. $G_{\mathrm{PS}}$-Invariant Tower")
ax.set_xlim(-5.2, 10.2)
ax.grid(True, axis="x", alpha=0.25)
ax.legend(loc="lower right", fontsize=8.0, frameon=True)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_quark_ckm_pulls_comparison.pdf"))
plt.close(fig)

# =============================================================================
# FIGURE 4: STRUCTURAL NO-GO SHIELDS (GENUINE REAL 10_H |r1|=1 & FRITZSCH NNI)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))

gens = [r"$y_e$ (1st gen)", r"$y_\mu$ (2nd gen)", r"$y_\tau$ (3rd gen)"]
ratios_real10 = v.results_json["nogo_shields"]["real_10_sv_ratios"]
x_g = np.arange(3)
ax1.bar(x_g - 0.17, ratios_real10, width=0.34, color="#d62728", alpha=0.85,
        label=r"Genuine real $\mathbf{10}_H$ ($|r_1|=1$) global min")
ax1.bar(x_g + 0.17, [1.0, 1.0, 1.0], width=0.34, color="#1f77b4", alpha=0.85,
        label=r"Complex $\mathbf{10}_H \oplus \overline{\mathbf{126}}_H$ (exact)")
ax1.axhline(1.0, color="black", lw=1.1, ls="--")
ax1.set_xticks(x_g)
ax1.set_xticklabels(gens)
ax1.set_ylabel(r"Singular-value ratio $y_\ell^{\mathrm{fit}}(M_R) / y_\ell^{\mathrm{target}}(M_R)$")
ax1.set_title(r"(a) Global Exclusion of Genuine Real $\mathbf{10}_H \oplus \overline{\mathbf{126}}_H$")
ax1.set_ylim(0.0, 2.15)
ax1.grid(True, axis="y", alpha=0.25)
ax1.legend(loc="upper right", fontsize=7.8, frameon=True)

bars_x = ["Fritzsch NNI\n$(1,3)=0$: $\\sqrt{y_u/y_c}$", "$K_4 \\times \\mathrm{Spin}(10)$\n$\\frac{3}{8}\\lambda_q$ ($M_R \\to M_Z$)", "PDG 2024\nObserved"]
bars_v = [v.nni_ratio, v.pred_MZ['Vub']/v.pred_MZ['Vcb'], v.vub_vcb_exp]
cols_b = ["#d62728", "#1f77b4", "#2ca02c"]
ax2.bar([0, 1, 2], bars_v, width=0.48, color=cols_b, alpha=0.88)
ax2.errorbar([2], [v.vub_vcb_exp], yerr=[v.vub_vcb_exp * np.hypot(0.00011/0.00369, 0.00085/0.04182)],
             fmt="none", ecolor="black", capsize=5, lw=1.5)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(bars_x)
ax2.set_ylabel(r"RG-invariant ratio $|V_{ub}| / |V_{cb}|$")
ax2.set_title(r"(b) Fritzsch NNI $(1,3)=0$ $1.98\times$ Shortfall Shield")
ax2.set_ylim(0.0, 0.108)
ax2.grid(True, axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_spin10_nogo_and_lee.pdf"))
plt.close(fig)

print("ALL 4 PAPER 25 v1.1.0 VECTOR PDF FIGURES GENERATED SUCCESSFULLY.")
