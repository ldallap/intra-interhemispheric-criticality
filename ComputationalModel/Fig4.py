"""
Reproduction of Companion Fig. 4 with our model implementation (consistent parameters).

  A  <rho> vs control parameter a (single population), with red/cyan/green example rasters for the
     synchronous / critical / asynchronous regimes
  B criticality anchor (Fano factor vs a) - C avalanches (P(S), tau) - D coupling drives integration (interhemi vs G)
  E-left   (a_I, a_II) grid, dots coloured by inter-population correlation (coolwarm)
  E-right  intra-population <rho> (Pop I vs Pop II), same colour coding; dashed diagonal a_I=a_II

Two sparsely coupled populations (p_between), weak coupling G=0.1, static a per population, no
shared input -- i.e. the original companion model (Eq. 3-4), in our implementation.

DATA / PLOT split:  python fig4.py data | plot | all     (SMOKE=1 for a fast layout check)
"""
from __future__ import annotations
import os, sys, time, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import powerlaw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from style import setup, save_panel, figsize_mm, INK, GUIDE

CACHE = HERE / "cache"                     # panel-data .pkl caches live beside the script
FIGURES = HERE.parent.parent / "figures"   # repo-root figures/ that the LaTeX includes
SMOKE = bool(os.environ.get("SMOKE"))
DATA = CACHE / ("fig4_data_smoke.pkl" if SMOKE else "fig4_data.pkl")
RED, CYAN, GREEN = "#E6034A", "#1CBEFF", "#00BFA0"          # sampled from the original Fig. 4
CMAP = "coolwarm"
# regime control-parameter values shown in Fig. 4 (must match the paper caption): synchronous /
# critical / asynchronous, straddling the firing cliff at a_crit~1.07 where <rho> falls ~0.6 -> 0
A_SYNC, A_CRIT, A_ASYNC = 1.02, 1.07, 1.12
RAS_BIN = 0.01                          # bin used to count spikes
BURN_SIM = 100.0 if SMOKE else 1000.0   # initial transient discarded from all <rho> estimates
# A)
T_A = 600.0 if SMOKE else 10000.0
T_RAS = 300.0 if SMOKE else 1100.0      # run longer; the first part is discarded as transient
RAS_BURN = 80 if SMOKE else 250         # fine bins (= sim-units) of raster transient to drop
As = np.round(np.linspace(1.00, 1.16, 17), 3)
# B)
T_ANCHOR = 1000.0 if SMOKE else 6000.0
# C)
T_AVAL = 800.0 if SMOKE else 5000.0
N_AVAL = 800 if SMOKE else 5000
RNG = np.random.default_rng(0)
# D)
T_G = 800.0 if SMOKE else 5000.0
Gs = np.array([0.0,0.5,1.0]) if SMOKE else np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
# E)
G_STAR = 0.3
NA = 5 if SMOKE else 13                 # finer grid to sample the narrow transition region
A_RANGE = np.round(np.linspace(1.00, 1.12, NA), 3)
T_E = 500.0 if SMOKE else 2500.0
# S9)
T_S = 1000.0 if SMOKE else 3500.0       # sim time for suppl fig 9


def _helpers(A):
    def post_burn(sc, b): # drop the initial synchronous transient
        nb = int(round(BURN_SIM / (b * 1000)))
        return sc[:, nb:]

    def rho_spikes(sc, b, cross=None, verbose=False):
        s = post_burn(sc, b); c = post_burn(cross, b) if cross is not None else None
        if cross is None:
            verbose and print(f"neurons with < 5 spikes: {np.sum(s.sum(axis=1) < 5)}/{s.shape[0]}")
        else:
            verbose and print(
                f"neurons with < 5 spikes: {np.sum(s.sum(axis=1) < 5)}/{s.shape[0]}, {np.sum(cross.sum(axis=1) < 5)}/{cross.shape[0]}")
        return A.rho_spikes(s, cross=c)

    def fire(sc, b):                                       # post-burn spikes / neuron / s
        s = post_burn(sc, b); return s.sum() / (s.shape[0] * s.shape[1] * b)

    return rho_spikes, fire


def compute_A(verbose=False):
    """Panel A: <rho> vs a (single population) + the three regime rasters. (fast)"""
    from simulator import simulate
    import analysis as A
    rho_spikes, fire = _helpers(A)
    rho_A = []
    ras_debug = []
    verbose and print(f"  progress (out of {len(As)}): ",end="",flush=True)
    for i, a in enumerate(As):
        verbose and print(f"{i}, ",end="",flush=True)
        r = simulate(K=250, T=T_A, G=0.0, g_in=0.0, p_between=0.0,
                     a_mode="static", a_value=float(a), spike_bin_sec=RAS_BIN, seed=1)
        b = r["spike_bin_sec"]
        rho_A.append(rho_spikes(r["sc0"], b) if fire(r["sc0"], b) > 5 else 0.0)   # silent -> 0
        ras_debug.append(r["sc0"])
    ras = {}
    for key, a in [("sync", A_SYNC), ("crit", A_CRIT), ("async", A_ASYNC)]:
        r = simulate(K=250, T=T_RAS, G=0.0, g_in=0.0, p_between=0.0, a_mode="static",
                     a_value=a, seed=2, spike_bin_sec=0.001)
        ras[key] = (r["sc0"][:, RAS_BURN:] > 0).astype(np.uint8)   # drop initial transient
    verbose and print()
    return {"A": (As, np.array(rho_A)), "ras": ras, "ras_debug": ras_debug,
            "regime_a": dict(sync=A_SYNC, crit=A_CRIT, async_=A_ASYNC)}


def compute_B():
    """Panel B: criticality anchor -- Fano-factor susceptibility of a single population vs `a` (peaks at a_crit)."""
    from simulator import simulate, SIM_PER_SEC
    import analysis as A
    av = np.array([1.00, 1.04, 1.07, 1.10, 1.13]) if SMOKE else np.round(np.arange(0.98, 1.141, 0.01), 3)
    chi_a = []
    for a in av:
        r = simulate(T=T_ANCHOR, G=0.0, g_in=0.0, a_mode="static", a_value=float(a), seed=1, record_every=20)
        _, fa = A.fano_windows(r["act0"], r["dt"], win_sec=T_ANCHOR/1000.0)
        chi_a.append(float(fa[0]) if len(fa) and np.isfinite(fa[0]) else 0.0)
    return {"SIM_PER_SEC": SIM_PER_SEC, "B": (av, np.array(chi_a))}


def compute_C():
    """Panel C: avalanche P(S) at a_crit, ~300-unit subsample (experiment protocol)."""
    from simulator import simulate
    import analysis as A
    r = simulate(K=N_AVAL, T=T_AVAL, dt=0.005, G=0.0, g_in=0.0, p_between=0.0,
                 a_mode="static", a_value=A_CRIT, seed=21, n_sub=min(300, N_AVAL // 2))
    nstep = r["asub0"].size; dT = max(1,int(round(nstep / max(1,r["asub0"].sum()))))
    S, _ = A.detect_avalanches(r["asub0"], dT=dT)
    Sf = S[(S >= 2) & (S <= N_AVAL)]
    if Sf.size > 40000:
        Sf = RNG.choice(Sf,40000,replace=False)
    tau = float(
        powerlaw.Fit(Sf,discrete=True,xmin=None,xmax=N_AVAL,verbose=False).power_law.alpha) if Sf.size > 100 else np.nan
    return {"C": (S, tau)}


def compute_D(verbose=False):
    """Panel D: interhemispheric correlation vs coupling `G`, for critical `a`."""
    from simulator import simulate
    import analysis as A
    rho_spikes, _ = _helpers(A)
    ic = np.full(len(Gs), np.nan)
    for i, G in enumerate(Gs):
        r = simulate(K=250, T=T_G, G=G, p_between=0.2, g_in=0.0, spike_bin_sec=RAS_BIN,
                     a_mode="static", a_value=A_CRIT, a_value1=A_CRIT, seed=3)
        b = r["spike_bin_sec"]
        verbose and print("  - G:",G,end=', ',flush=True)
        ic[i] = rho_spikes(r["sc0"], b, cross=r["sc1"], verbose=verbose)
    return {"D": (Gs, ic)}


def compute_E(verbose=False):
    """Panel E: (a_I, a_II) grid of weakly all-to-all coupled populations. (slow)"""
    from simulator import simulate
    import analysis as A
    rho_spikes, _ = _helpers(A)
    inter = np.full((NA,NA), np.nan); rI = np.full((NA,NA), np.nan); rII = np.full((NA,NA), np.nan)
    verbose and print(f"  progress (out of {len(A_RANGE)}): ",end="",flush=True)
    for i, aI in enumerate(A_RANGE):
        verbose and print(f"{i}, ",end="",flush=True)
        for j, aII in enumerate(A_RANGE):
            r = simulate(K=250, T=T_E, G=G_STAR, p_between=0.2, g_in=0.0, spike_bin_sec=RAS_BIN,
                         a_mode="static", a_value=float(aI), a_value1=float(aII), seed=3)
            b = r["spike_bin_sec"]
            rI[i,j] = rho_spikes(r["sc0"],b)
            rII[i,j] = rho_spikes(r["sc1"],b)
            inter[i,j] = rho_spikes(r["sc0"],b,cross=r["sc1"])
    verbose and print()
    return {"E": (A_RANGE, inter, rI, rII)}


def compute_Supp9(verbose=False):
    """Supplementary Figure 9: (a_I, a_II) grid of uncoupled populations."""
    from simulator import simulate
    import analysis as A
    rho_spikes, _ = _helpers(A)
    # sim
    r = []
    for i in range(0, len(A_RANGE)):
        r.append(simulate(K=250, T=T_S, G=0.0, aa_between=True, g_in=0.0, spike_bin_sec=RAS_BIN,
                          a_mode="static", a_value=float(A_RANGE[i]), a_value1=float(A_RANGE[i]), seed=3))
        i == 0 and verbose and print(f"  bin: {r[-1]["spike_bin_sec"]}, spike matrix shape: {r[-1]["sc0"].shape}\n  a: {A_RANGE[i]}",end="",flush=True)
        i != 0 and verbose and print(f", {A_RANGE[i]}",end="",flush=True)
    verbose and print()
    # correlation
    inter = np.full((NA,NA),np.nan); rI = np.full((NA,NA),np.nan); rII = np.full((NA,NA),np.nan)
    for i, aI in enumerate(A_RANGE):
        b = r[i]["spike_bin_sec"]; ri = r[i]["sc0"]
        for j, aII in enumerate(A_RANGE):
            rj = r[j]["sc1"]
            rI[i,j] = rho_spikes(ri,b)
            rII[i,j] = rho_spikes(rj,b)
            inter[i,j] = rho_spikes(ri,b,cross=rj)
    return {"S9": (A_RANGE, inter, rI, rII)}


def draw(d):
    setup()
    fig = plt.figure(figsize=figsize_mm(170, 230))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.05, 1.0, 0.08, 0.9], width_ratios=[1, 1, 1],
                          hspace=0.85, wspace=0.3)

    a_A, rho_A = d["A"]; ras = d["ras"]; ra = d["regime_a"]

    # ---- A (spans top row) ----
    axA = fig.add_subplot(gs[0, :])
    axA.plot(a_A, rho_A, "-", color=GUIDE, lw=1.0, zorder=1)
    axA.plot(a_A, rho_A, "o", mfc="white", mec=GUIDE, ms=4, zorder=2)
    reg = [("sync", ra["sync"], RED), ("crit", ra["crit"], CYAN), ("async", ra["async_"], GREEN)]
    # red & cyan low (cyan sits over the flat-zero region, off the curve), green higher and right
    insets = {"sync": [0.02, 0.06, 0.23, 0.37], "crit": [0.30, 0.05, 0.22, 0.36],
              "async": [0.72, 0.50, 0.24, 0.42]}
    for key, a, col in reg:
        rho_here = float(np.interp(a, a_A, rho_A))
        axA.plot([a], [rho_here], "o", color=col, ms=8, zorder=3)
        ax_in = axA.inset_axes(insets[key])
        R = ras[key]
        if key in ("sync", "crit"):
            R = R[:, :R.shape[1] // 2] # dense regimes: show half the time window
        ys, xs = np.nonzero(R)
        ax_in.scatter(xs, ys, s=2.2, marker="o", color=col, linewidths=0)
        ax_in.set_xticks([]); ax_in.set_yticks([]); ax_in.set_xlim(0, R.shape[1])
        for s in ax_in.spines.values():
            s.set_edgecolor(col); s.set_linewidth(0.9)
    axA.set(xlabel="control parameter $a$",ylabel=r"mean correlation $\langle\rho\rangle$")
    axA.set_ylim(min(-0.05, np.nanmin(rho_A)), np.nanmax(rho_A)*1.15)

    # ---- B: criticality anchor ----
    av, chi_a = d["B"]
    axB = fig.add_subplot(gs[1, 0])
    axB.plot(av, chi_a, "o-", color="gray", ms=3)
    axB.axvline(A_CRIT, ls="--", color=INK, lw=0.8)
    axB.annotate(r"$a_\mathrm{crit}\approx1.07$", xy=(A_CRIT, np.nanmax(chi_a)),
                 xytext=(A_CRIT-0.06, np.nanmax(chi_a)*0.82), fontsize=7, color=INK)
    axB.set(xlabel="excitability $a$",ylabel="Fano factor (susceptibility)",box_aspect=1)
    axB.set_title("Criticality anchor", loc="left")

    # ---- C: avalanche P(S) at a_crit ----
    S, tau = d["C"]
    axC = fig.add_subplot(gs[1, 1])
    vals, cnts = np.unique(S[S >= 1],return_counts=True); p = cnts / cnts.sum()
    axC.loglog(vals,p,"o",color=CYAN,ms=2.5)
    xr = np.array([2.0,max(3.0,float(vals.max()))]); p2 = p[vals == 2][0] if (vals == 2).any() else p[0]
    axC.loglog(xr,p2 * (xr / 2.0) ** (-tau),"--",color=INK,lw=0.9)
    axC.text(0.55,0.9,fr"$\tau\approx{tau:.2f}$",transform=axC.transAxes,ha="center")
    axC.set(xlabel="avalanche size $S$",ylabel="$P(S)$",box_aspect=1)
    axC.set_title("Neuronal avalanches",loc="left")

    # ---- D: interhemispheric correlation vs G ----
    Gs, ic = d["D"]
    axD = fig.add_subplot(gs[1, 2])
    axD.plot(Gs, ic, "o-", color="gray", ms=3); axD.axvline(G_STAR, ls=":", color=GUIDE, lw=0.8)
    axD.set(xlabel="coupling $G$",ylabel="interhemispheric corr",box_aspect=1)
    axD.set_title("Coupling drives integration",loc="left")

    # ---- E-left: (a_I, a_II) ----
    A_RANGE, inter, rI, rII = d["E"]
    cax = fig.add_subplot(gs[2,0:2])
    axEL = fig.add_subplot(gs[3, 0])
    AI, AII = np.meshgrid(A_RANGE, A_RANGE, indexing="ij")
    axEL.scatter(AI.ravel(), AII.ravel(), c=inter.ravel(), cmap=CMAP, vmin=-0.3, vmax=0.3, s=42, edgecolors="none")
    lim = [A_RANGE.min(), A_RANGE.max()]; axEL.plot(lim, lim, "--", color=INK, lw=0.7)
    axEL.set(xlabel=r"$a$ Pop. I",ylabel=r"$a$ Pop. II",aspect="equal")

    # ---- E-right: (rho_I, rho_II) ----
    axER = fig.add_subplot(gs[3, 1])
    sc = axER.scatter(rI.ravel(), rII.ravel(), c=inter.ravel(), cmap=CMAP, vmin=-0.3, vmax=0.3, s=24, edgecolors="none")
    m = max(np.nanmax(rI), np.nanmax(rII)); axER.plot([0, m], [0, m], "--", color=INK, lw=0.7)
    axER.set(xlabel=r"Pop. I $\langle\rho\rangle$",ylabel=r"Pop. II $\langle\rho\rangle$",aspect="equal")

    cb = fig.colorbar(sc, cax=cax, orientation="horizontal",location="top")
    cb.set_label("mean correlation inter-population", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # panel letters A, B, C at the same figure x (aligned left edges)
    xL = axA.get_position().x0 - 0.045
    fig.text(xL, axA.get_position().y1, "A", fontweight="bold", fontsize=12, ha="left", va="bottom")
    fig.text(xL, axB.get_position().y1,"B",fontweight="bold", fontsize=12, ha="left", va="bottom")
    fig.text(axC.get_position().x0 - 0.045, axC.get_position().y1,"C",fontweight="bold", fontsize=12, ha="left", va="bottom")
    fig.text(axD.get_position().x0 - 0.045, axD.get_position().y1,"D",fontweight="bold", fontsize=12, ha="left", va="bottom")
    fig.text(xL, axEL.get_position().y1, "E", fontweight="bold", fontsize=12, ha="left", va="bottom")

    save_panel(fig, FIGURES / "fig4_repro")


def draw_supp(d):
    setup()
    A_RANGE, inter, rI, rII = d["S9"]

    fig = plt.figure(figsize=figsize_mm(120, 80))
    gs = fig.add_gridspec(1, 2, width_ratios=[1,1], hspace=0.45, wspace=0.4)

    # ---- left: (a_I, a_II) ----
    axBL = fig.add_subplot(gs[0,0])
    AI, AII = np.meshgrid(A_RANGE, A_RANGE, indexing="ij")
    axBL.scatter(AI.ravel(), AII.ravel(), c=inter.ravel(), cmap=CMAP, vmin=-0.06, vmax=0.06, s=42, edgecolors="none")
    lim = [A_RANGE.min(), A_RANGE.max()]; axBL.plot(lim, lim, "--", color=INK, lw=0.7)
    axBL.set_xlabel(r"$a$ Pop. I"); axBL.set_ylabel(r"$a$ Pop. II")
    axBL.set_aspect("equal")

    # ---- right: (rho_I, rho_II) ----
    axBR = fig.add_subplot(gs[0,1])
    sc = axBR.scatter(rI.ravel(), rII.ravel(), c=inter.ravel(), cmap=CMAP, vmin=-0.06, vmax=0.06, s=24, edgecolors="none")
    m = max(np.nanmax(rI), np.nanmax(rII)); axBR.plot([0, m], [0, m], "--", color=INK, lw=0.7)
    axBR.set_xlabel(r"Pop. I $\langle\rho\rangle$"); axBR.set_ylabel(r"Pop. II $\langle\rho\rangle$")
    axBR.set_aspect("equal")

    cb = fig.colorbar(sc, ax=[axBL, axBR], orientation="horizontal", fraction=0.05, pad=0.18,
                      location="top", aspect=40)
    cb.set_label("mean correlation inter-population", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    save_panel(fig, FIGURES / "supp_fig9_uncoupled")


def main():
    # modes: data (main fig) | dataA (panel A only; same for B, C) | dataSupp (supplementary only) | plot | plotSupp | all
    verbose = "-v" in sys.argv[1:]
    args = [arg for arg in sys.argv[1:] if arg != "-v"]
    mode = args[0] if args else "plot"
    t0 = time.time()
    load = lambda: pickle.load(open(DATA, "rb"))
    save = lambda d: pickle.dump(d, open(DATA, "wb"))
    if mode in ("data", "all"):
        print(f"SMOKE={SMOKE}; generating fig4 data ...")
        save({**compute_A(verbose), **compute_B(), **compute_C(), **compute_D(verbose), **compute_E(verbose)})
        print(f"  saved {DATA.name} ({time.time()-t0:.0f}s)")
    elif mode == "dataA":
        d = load(); d.update(compute_A(verbose)); save(d); print(f"  updated panel A ({time.time()-t0:.0f}s)")
    elif mode == "dataB":
        d = load(); d.update(compute_B()); save(d); print(f"  updated panel B ({time.time()-t0:.0f}s)")
    elif mode == "dataC":
        d = load(); d.update(compute_C()); save(d); print(f"  updated panel C ({time.time()-t0:.0f}s)")
    elif mode == "dataD":
        d = load(); d.update(compute_D(verbose)); save(d); print(f"  updated panel D ({time.time()-t0:.0f}s)")
    elif mode == "dataE":
        d = load(); d.update(compute_E(verbose)); save(d); print(f"  updated panel E ({time.time()-t0:.0f}s)")
    elif mode == "dataSupp":
        d = load(); d.update(compute_Supp9(verbose)); save(d); print(f"  updated fig S9 ({time.time()-t0:.0f}s)")
    if mode in ("plot", "all"):
        if not DATA.exists():
            sys.exit(f"no {DATA.name}; run `python fig4.py data` first")
        draw(load()); print(f"  drawn ({time.time()-t0:.0f}s)")
    if mode == "plotSupp":
        draw_supp(load()); print(f"  drawn fig S9 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
