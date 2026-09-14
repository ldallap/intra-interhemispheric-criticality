"""
Master story figure for the companion-paper model revision.

DATA / PLOT split so plot tweaks never re-run the simulations:
    python Figure_story.py data    # run all sims ONCE -> figure_story_data.pkl  (slow, ~15 min)
    python Figure_story.py plot    # load the .pkl and draw the figure           (instant)
    python Figure_story.py all     # data + plot in one go
    SMOKE=1 python Figure_story.py all   # tiny settings, validate layout fast

Panels:
  A fluctuations around criticality: chi(t)+a(t)
  B driver signature (inverted-U chi vs a-a_crit) - C ablation - D correlated slow-drive trajectory in (a1,a2)
"""
from __future__ import annotations
import os, sys, time, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import powerlaw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # style.py
sys.path.insert(0, str(HERE.parent))          # simulator.py, analysis.py
from style import setup, save_panel, figsize_mm, INK, GUIDE, categorical, ACCENT_WARM, ACCENT_COOL

SMOKE = bool(os.environ.get("SMOKE"))
CACHE = HERE / "cache"                      # panel-data .pkl caches live beside the script
OUT = HERE.parent.parent / "figures"        # repo-root figures/ that the LaTeX includes
DATA = CACHE / ("figure_story_data_smoke.pkl" if SMOKE else "figure_story_data.pkl")
A_CRIT, WIN = 1.07, 1.0                 # critical point = synchronization transition (Fano peak)
G_STAR = 0.3
OP = dict(K=250, p_between=0.2, a_mu=A_CRIT, a_s=0.06, tau_a_sec=6.0,
          in_fluct=0.4, tau_in_sec=6.0, record_every=20)
T_ABL = 6000.0 if SMOKE else 50000.0    # long: Fano needs many windows


# ----------------------------------------------------------------- data generation (slow)
def chi_series(r):
    """Per-window Fano-factor susceptibility (peaks at criticality) + mean drive a per window."""
    import analysis as A
    _, fano = A.fano_windows(r["act0"], r["dt"], WIN)
    win = max(5, int(round(WIN * 1000.0 / (r["record_every"]*r["dt"]))))
    n = r["a0"].size // win
    aw = np.array([r["a0"][w*win:(w+1)*win].mean() for w in range(n)])
    m = min(len(fano), len(aw))
    return fano[:m], aw[:m]


def compute():
    from simulator import simulate, SIM_PER_SEC
    import analysis as A
    d = {"SIM_PER_SEC": SIM_PER_SEC}

    # ablation (FULL reused for C/D/G panels)
    configs = {
        "FULL": dict(a_mode="ou", a_corr=0.7, g_in=0.2, G=G_STAR),
        "−dyn a": dict(a_mode="static", a_value=A_CRIT, g_in=0.2, G=G_STAR),
        "−input": dict(a_mode="ou", a_corr=0.7, g_in=0.0, G=G_STAR),
        "−coupling": dict(a_mode="ou", a_corr=0.7, g_in=0.2, G=0.0),
        "−a corr": dict(a_mode="ou", a_corr=0.0, g_in=0.2, G=G_STAR),
    }
    std_chi, full = {}, None
    for name, cfg in configs.items():
        rr = simulate(T=T_ABL, seed=11, **OP, **cfg)
        chi, aw = chi_series(rr)
        std_chi[name] = float(np.nanstd(chi))
        if name == "FULL":
            full = rr
    d["F_abl"] = std_chi
    ctr, chi = A.fano_windows(full["act0"], full["dt"], WIN)
    chi_f, aw_f = chi_series(full)
    ok = np.isfinite(chi_f)
    ccE = float(np.corrcoef(np.abs(aw_f[ok] - A_CRIT), chi_f[ok])[0, 1]) if ok.sum() > 3 else np.nan
    d["D"] = (ctr, chi, full["t_rec"]/SIM_PER_SEC, full["a0"], full["a1"])
    d["E"] = (aw_f - A_CRIT, chi_f, ccE)
    d["G"] = (full["a0"], full["a1"], full["t_rec"]/SIM_PER_SEC, float(np.corrcoef(full["a0"], full["a1"])[0, 1]))

    return d


# ----------------------------------------------------------------- plotting (fast)
def _smooth(x, k=3):
    """NaN-aware centered moving average (k windows) to show the seconds-scale envelope."""
    x = np.asarray(x, float); out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = x[max(0, i-k//2):i+k//2+1]; seg = seg[np.isfinite(seg)]
        out[i] = seg.mean() if len(seg) else np.nan
    return out


def draw(d):
    setup()
    SP = d["SIM_PER_SEC"]
    fig = plt.figure(figsize=figsize_mm(174, 105))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.95, 1], hspace=0.6, wspace=0.42)
    cat = categorical(6)

    # A (wide): Fano susceptibility over time + the slow drive crossing a_crit
    axD = fig.add_subplot(gs[0, :]); ctr, chi, ts, a0, a1 = d["D"]
    axD.plot(ctr, _smooth(chi, 3), "-", color=cat[0], lw=1.8, label=r"Fano susceptibility $\chi(t)$")
    axD.set_ylabel(r"Fano susceptibility $\chi(t)$"); axD.set_xlabel("time (s)")
    axD.set_title("Fluctuations around criticality on a seconds timescale (driven by the slow excitability)", loc="left")
    axR = axD.twinx(); axR.spines["top"].set_visible(False)
    axR.plot(ts, a0, color=INK, lw=0.7, alpha=0.85, label=r"drive $a(t)$")
    axR.axhline(A_CRIT, ls="--", color=ACCENT_WARM, lw=1.0, label=r"$a_\mathrm{crit}=1.07$")
    axR.set_ylabel(r"drive $a(t)$")
    h1, l1 = axD.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
    axD.legend(h1+h2, l1+l2, loc="upper right", ncol=3, fontsize=6, frameon=False)

    # B: susceptibility vs distance from criticality -> inverted-U (binned mean)
    axE = fig.add_subplot(gs[1, 0]); da, chf, cc = d["E"]
    ok = np.isfinite(chf)
    axE.scatter(da[ok], chf[ok], s=10, color=cat[3])
    axE.axvline(0, ls="--", color=ACCENT_WARM, lw=1.0)
    axE.set_xlabel(r"$a - a_\mathrm{crit}$"); axE.set_ylabel(r"Fano susceptibility $\chi$")
    axE.set_title(r"Driver: $\chi$ peaks at $a_\mathrm{crit}$", loc="left")

    # C
    axF = fig.add_subplot(gs[1, 1]); names = list(d["F_abl"]); vals = [d["F_abl"][n] for n in names]
    floor = d["F_abl"].get("−dyn a", np.nan)
    # FULL highlighted; the static-a bar IS the floor (no dynamic drive)
    cols = [cat[0]] + [GUIDE if n == "−dyn a" else cat[4] for n in names[1:]]
    axF.bar(range(len(names)), vals, color=cols)
    axF.axhline(floor, ls="--", color=INK, lw=0.8)
    axF.text(len(names)-0.4, floor, "  static-$a$ floor\n  (no drive)", fontsize=5.5, va="center", ha="right", color=INK)
    axF.set_xticks(range(len(names))); axF.set_xticklabels(names, rotation=40, ha="right", fontsize=6)
    axF.set_ylabel(r"fluctuation amplitude  std($\chi$)"); axF.set_title("What drives the fluctuations", loc="left")

    # D
    axG = fig.add_subplot(gs[1, 2]); a0, a1, tt, rr = d["G"]
    axG.scatter(a0, a1, c=tt, s=3, cmap="brightPMY")
    axG.axvline(A_CRIT, ls=":", color=GUIDE, lw=0.7); axG.axhline(A_CRIT, ls=":", color=GUIDE, lw=0.7)
    axG.set_xlabel(r"$a_1$"); axG.set_ylabel(r"$a_2$"); axG.set_title(f"Drive trajectory (r={rr:.2f})", loc="left")

    for ax, L in [(axD, "A"), (axE, "B"), (axF, "C"), (axG, "D")]:
        ax.text(-0.02, 1.2, L, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top", ha="right")

    save_panel(fig, OUT / "Figure_story")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plot"
    t0 = time.time()
    if mode in ("data", "all"):
        print(f"SMOKE={SMOKE}; generating data ...")
        d = compute()
        with open(DATA, "wb") as f:
            pickle.dump(d, f)
        print(f"  saved {DATA.name} ({time.time()-t0:.0f}s)")
    if mode in ("plot", "all"):
        if not DATA.exists():
            sys.exit(f"no data file {DATA.name}; run `python Figure_story.py data` first")
        with open(DATA, "rb") as f:
            d = pickle.load(f)
        draw(d)
        print(f"  drawn from {DATA.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
