"""
Analysis for the two-population active-rotator prototype.

NOTE ON SYNCHRONY AND CRITICALITY MEASURES
------------------------------------------
Synchrony is measured by <rho> (rho_windows): the mean pairwise correlation of (Mexican-hat-)
kernel-convolved spike trains, the published companion measure -- this is what the paper's fig4.py
uses. The Kuramoto order parameter R_n(t) is also recorded as the natural model observable
(monotonically related to <rho>), but it is a DESCRIPTOR only.

The criticality LOCATOR is the FANO FACTOR (fano_windows), which peaks at the synchronization
transition a_crit~1.07. Var(R) is NOT a reliable locator here: |Z| stays large in the asynchronous
regime, so Var(R) peaks OFF the transition (near a~1.02) -- see the CAUTION on indicator_windows,
which is kept only for the earlier-calibration exploration scripts. The avalanche / DCC code is
independent of the synchrony-axis choice.
"""

from __future__ import annotations
import numpy as np

try:
    import powerlaw  # MLE heavy-tail fits, as used in the paper (Alstott et al.)
    _HAVE_POWERLAW = True
except Exception:
    _HAVE_POWERLAW = False


# ----------------------------------------------------------------------------- synchrony / coupling
def window_means(x, t, win_sec, sim_per_sec=1000.0):
    """Mean of a downsampled signal x(t) over non-overlapping windows of win_sec real seconds."""
    win = win_sec*sim_per_sec
    if t.size < 2:
        return np.array([]), np.array([])
    edges = np.arange(t[0], t[-1], win)
    idx = np.searchsorted(edges, t) - 1
    nb = max(idx.max() + 1, 0)
    out = np.array([x[idx == b].mean() for b in range(nb) if np.any(idx == b)])
    ctr = np.array([edges[b] + win/2 for b in range(nb) if np.any(idx == b)])
    return ctr, out


def interhemispheric_correlation(act0, act1, dt, win_sec=0.05, sim_per_sec=1000.0):
    """Pearson r between the two populations' binned activity (proxy for interhemispheric coupling).

    Binned at win_sec to suppress single-spike noise; this is the model proxy for the paper's
    interhemispheric pairwise correlation. Returns a single scalar r over the whole run.
    """
    bw = max(1, int(round(win_sec*sim_per_sec/dt)))
    n = (act0.size//bw)*bw
    b0 = act0[:n].reshape(-1, bw).sum(1).astype(float)
    b1 = act1[:n].reshape(-1, bw).sum(1).astype(float)
    if b0.std() == 0 or b1.std() == 0:
        return np.nan
    return float(np.corrcoef(b0, b1)[0, 1])


def fano_windows(act, dt, win_sec, bin_sec=0.02, sim_per_sec=1000.0, min_mean=2.0):
    """Fano factor (variance / mean) of the population activity, per window. This is the
    avalanche/branching susceptibility: it peaks at the critical point (the synchronization
    transition), unlike Var(|Z|) which is fooled by the silent-high-R artifact. `act` is the per-step
    population spike count. Windows whose mean activity is below min_mean return NaN (silent)."""
    bw = max(1, int(round(bin_sec*sim_per_sec/dt)))
    n = (act.size // bw)*bw
    A = act[:n].reshape(-1, bw).sum(1).astype(float)        # binned population activity
    bpw = max(2, int(round(win_sec/bin_sec)))
    ctr, fano = [], []
    for w in range(A.size // bpw):
        seg = A[w*bpw:(w+1)*bpw]; m = seg.mean()
        fano.append(seg.var()/m if m > min_mean else np.nan)
        ctr.append((w + 0.5)*bpw*bin_sec)
    return np.array(ctr), np.array(fano)


# ---------------------------------------------- finite-size-robust criticality indicators (per window)
def indicator_windows(R, dt_rec_sim, win_sec, sim_per_sec=1000.0):
    """Var(R) and lag-1 autocorrelation AC1 of the order parameter, per window. LEGACY: used only by
    the earlier-calibration exploration scripts, NOT by the paper figures or tutorial.

    CAUTION: Var(R) is NOT the criticality locator for this model. |Z| stays large in the
    asynchronous regime, so Var(R) peaks OFF the transition (near a~1.02), not at a_crit~1.07. Use
    fano_windows (the Fano factor) to locate criticality; see the module note above. AC1 is a
    critical-slowing-down descriptor. R is the downsampled order parameter sampled every dt_rec_sim
    simulation-time units. Returns (window_centers_sec, chi, ac1).
    """
    win = max(5, int(round(win_sec*sim_per_sec/dt_rec_sim)))
    n = R.size // win
    ctr, chi, ac1 = [], [], []
    for w in range(n):
        seg = R[w*win:(w+1)*win]
        chi.append(float(seg.var()))
        s = seg - seg.mean(); v = float((s*s).mean())
        ac1.append(float((s[:-1]*s[1:]).mean()/v) if v > 0 else np.nan)
        ctr.append((w + 0.5)*win*dt_rec_sim/sim_per_sec)
    return np.array(ctr), np.array(chi), np.array(ac1)


# --------------------------------------------------------------- <rho>: kernel pairwise spike corr
def _mexican_hat(bin_sec, s1_sec=0.1, s2_sec=0.4, truncate=3.0):
    """Difference-of-Gaussians kernel (companion: 100 ms - 400 ms) on the bin grid."""
    s1, s2 = s1_sec/bin_sec, s2_sec/bin_sec
    x = np.arange(-int(np.ceil(truncate*s2)), int(np.ceil(truncate*s2)) + 1)
    g = lambda s: np.exp(-0.5*(x/s)**2)/(s*np.sqrt(2*np.pi))
    return g(s1) - g(s2)


def rho_windows(sc, bin_sec, win_sec, cross=None):
    """Mean pairwise correlation <rho> of kernel-convolved spike trains, per window (the companion's
    synchrony measure). sc: (K, nbins) per-neuron binned spikes. If `cross` (another (K,nbins)) is
    given, returns the mean CROSS-population (interhemispheric) correlation instead of within-region.
    Returns (window_centers_sec, rho_per_window).
    """
    k = _mexican_hat(bin_sec)
    conv = np.array([np.convolve(r, k, mode="same") for r in sc.astype(float)])
    convx = (np.array([np.convolve(r, k, mode="same") for r in cross.astype(float)])
             if cross is not None else None)
    bpw = max(1, int(round(win_sec/bin_sec)))
    n_win = conv.shape[1] // bpw
    ctr, rho = [], []
    for w in range(n_win):
        seg = conv[:, w*bpw:(w+1)*bpw]
        a = seg[seg.std(1) > 0]
        if cross is None:
            if a.shape[0] < 3:
                rho.append(np.nan)
            else:
                C = np.corrcoef(a); rho.append(np.nanmean(C[np.triu_indices(C.shape[0], 1)]))
        else:
            seg2 = convx[:, w*bpw:(w+1)*bpw]; b = seg2[seg2.std(1) > 0]
            if a.shape[0] < 3 or b.shape[0] < 3:
                rho.append(np.nan)
            else:
                A = (a - a.mean(1, keepdims=True))/a.std(1, keepdims=True)
                B = (b - b.mean(1, keepdims=True))/b.std(1, keepdims=True)
                rho.append(float(np.nanmean((A @ B.T)/A.shape[1])))
        ctr.append((w + 0.5)*bpw*bin_sec)
    return np.array(ctr), np.array(rho)


def rho_spikes(sc, cross=None):
    """Mean pairwise correlation <rho> of raw spike trains. sc: (K, nbins) per-neuron binned spikes.
    If `cross` (another (K, nbins)) is given, returns the mean CROSS-population (interhemispheric)
    correlation instead of within-region. Returns (window_centers_sec, rho_per_window).
    """
    rho = np.nan
    s = sc[sc.std(1) > 0]
    if cross is None:
        if s.shape[0] >= 3:
            C = np.corrcoef(s); rho = np.nanmean(C[np.triu_indices(C.shape[0], 1)])
    else:
        c = cross[cross.std(1) > 0]
        if s.shape[0] >= 3 and c.shape[0] >= 3:
            A = (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)
            B = (c - c.mean(1, keepdims=True)) / c.std(1, keepdims=True)
            rho = np.nanmean((A @ B.T) / A.shape[1])

    return float(rho)


# ----------------------------------------------------------------------------------- avalanches
def detect_avalanches(act, dT=None, return_spans=False):
    """Avalanches from a per-step population spike-count series.

    Bin width = mean population interspike interval (DeltaT), following Beggs & Plenz / the paper.
    Pass dT explicitly (in steps) to use a fixed bin across windows; otherwise it is estimated from
    `act`. An avalanche is a maximal run of consecutive non-empty bins. Returns (sizes, durations)
    and, optionally, the (start_bin, end_bin, bin_width_steps) spans for synchrony grouping.
    """
    total = int(act.sum())
    if total < 10:
        empty = (np.array([]), np.array([]))
        return (empty + ([],)) if return_spans else empty
    if dT is None:
        dT = max(1, int(round(act.size/total)))      # mean population ISI in steps
    n = (act.size//dT)*dT
    binned = act[:n].reshape(-1, dT).sum(1)
    occupied = binned > 0

    sizes, durations, spans = [], [], []
    i, nb = 0, binned.size
    while i < nb:
        if occupied[i]:
            j = i
            while j < nb and occupied[j]:
                j += 1
            sizes.append(int(binned[i:j].sum()))
            durations.append(j - i)
            spans.append((i, j, dT))
            i = j
        else:
            i += 1
    sizes = np.asarray(sizes); durations = np.asarray(durations)
    return (sizes, durations, spans) if return_spans else (sizes, durations)


def _fit_exponent(data, xmin, xmax):
    """Power-law exponent (alpha) via MLE; powerlaw package if available, else discrete MLE."""
    data = np.asarray(data)
    data = data[(data >= xmin) & (data <= xmax)]
    if data.size < 20:
        return np.nan
    if _HAVE_POWERLAW:
        try:
            fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax, discrete=True, verbose=False)
            return float(fit.power_law.alpha)
        except Exception:
            pass
    # Clauset discrete-MLE fallback
    return float(1 + data.size/np.sum(np.log(data/(xmin - 0.5))))


def dcc(sizes, durations, smin=3, smax=38, tmin=1, tmax=18):
    """Distance to Criticality Coefficient.

    tau (size), tau_t (duration) from power-law fits; predicted scaling exponent (tau_t-1)/(tau-1)
    vs the empirical 1/(sigma nu z) from <S>(T) ~ T^{1/sigma nu z}. DCC = |empirical - predicted|.
    Default bounds match the paper (size 3-38, duration 1-18).
    """
    tau = _fit_exponent(sizes, smin, smax)
    tau_t = _fit_exponent(durations, tmin, tmax)
    if not np.isfinite(tau) or not np.isfinite(tau_t) or tau <= 1:
        return dict(tau=tau, tau_t=tau_t, gamma_pred=np.nan, gamma_emp=np.nan, dcc=np.nan)
    gamma_pred = (tau_t - 1)/(tau - 1)
    # empirical <S>(T): mean size per duration, log-log slope
    durs = np.asarray(durations); szs = np.asarray(sizes)
    m = (durs >= tmin) & (durs <= tmax)
    durs, szs = durs[m], szs[m]
    uniq = np.unique(durs)
    if uniq.size < 3:
        gamma_emp = np.nan
    else:
        mean_s = np.array([szs[durs == d].mean() for d in uniq])
        good = mean_s > 0
        gamma_emp = float(np.polyfit(np.log(uniq[good]), np.log(mean_s[good]), 1)[0])
    return dict(tau=tau, tau_t=tau_t, gamma_pred=gamma_pred, gamma_emp=gamma_emp,
                dcc=abs(gamma_emp - gamma_pred) if np.isfinite(gamma_emp) else np.nan)


def dcc_vs_synchrony(act, R, t_rec, dt, n_bins=12):
    """Reproduce Fig. 2A-B logic: group avalanches by the synchrony level (mean R) during each
    avalanche, then compute DCC per synchrony bin. Returns (sync_centers, dcc_values, counts).
    """
    sizes, durations, spans = detect_avalanches(act, return_spans=True)
    if len(spans) < 50:
        return np.array([]), np.array([]), np.array([])
    # mean R during each avalanche (map bin span -> step -> nearest recorded R sample)
    rec_step = (t_rec/dt)
    av_sync = np.empty(len(spans))
    for k, (i, j, dT) in enumerate(spans):
        s0, s1 = i*dT, j*dT
        lo = np.searchsorted(rec_step, s0); hi = max(lo + 1, np.searchsorted(rec_step, s1))
        av_sync[k] = R[lo:hi].mean() if hi <= R.size else R[lo:].mean()

    edges = np.linspace(np.nanmin(av_sync), np.nanmax(av_sync), n_bins + 1)
    centers, dccs, counts = [], [], []
    for b in range(n_bins):
        m = (av_sync >= edges[b]) & (av_sync < edges[b + 1])
        if m.sum() < 50:
            continue
        d = dcc(sizes[m], durations[m])
        centers.append((edges[b] + edges[b + 1])/2)
        dccs.append(d['dcc']); counts.append(int(m.sum()))
    return np.array(centers), np.array(dccs), np.array(counts)


def dcc_per_window(act, dt, win_sec, t_rec=None, a_rec=None, sim_per_sec=1000.0, min_aval=60):
    """DCC computed in consecutive TIME windows -> DCC(t): the temporal 'fluctuations around
    criticality' (the system hovering in/out of the critical region over time).

    Uses a single global DeltaT so window-to-window changes reflect dynamics, not binning. Windows
    with < min_aval avalanches return NaN (the silent/subcritical extreme, as in the paper). If
    a_rec/t_rec are given, also returns the mean latent drive a per window -> the mechanistic link.
    """
    total = int(act.sum())
    if total < 1000:
        return dict(t=np.array([]), dcc=np.array([]), n_aval=np.array([]), a_win=None, dT=0)
    dT = max(1, int(round(act.size/total)))
    win_steps = int(round(win_sec*sim_per_sec/dt))
    n_win = act.size // win_steps
    have_a = a_rec is not None and t_rec is not None
    rec_step = (t_rec/dt) if have_a else None
    t, dccs, nav, aw = [], [], [], []
    for w in range(n_win):
        seg = act[w*win_steps:(w+1)*win_steps]
        s, d = detect_avalanches(seg, dT=dT)
        t.append((w + 0.5)*win_steps*dt)
        dccs.append(dcc(s, d)["dcc"] if len(s) >= min_aval else np.nan)
        nav.append(len(s))
        if have_a:
            lo = np.searchsorted(rec_step, w*win_steps); hi = np.searchsorted(rec_step, (w+1)*win_steps)
            aw.append(a_rec[lo:hi].mean() if hi > lo else np.nan)
    return dict(t=np.array(t), dcc=np.array(dccs), n_aval=np.array(nav),
                a_win=np.array(aw) if have_a else None, dT=dT, win_steps=win_steps)
