"""
Two-population active-rotator simulator (companion-paper revision prototype, v2).

Model = Shinomoto-Kuramoto / Buendia et al. PRResearch 3, 023224 (2021), Eq. (3); companion Eq. (3-4):

    dphi_i = [ omega + a_n(t)*sin(phi_i)
               + J * (1/K)  Sum_{j in n}      sin(phi_j - phi_i)    # WITHIN region: all-to-all (mean field)
               + G * (1/K)  Sum_{j in A_{n<-m}} sin(phi_j - phi_i)   # BETWEEN regions: SPARSE (density p_between)
               + g_in * F(t) * sin(theta(t) - phi_i) ] dt           # COMMON INPUT (rest-of-network), global scale g_in
             + sigma * dW_i

Verified against Buendia 2021: J = 1, omega = 1, HT critical point at a = 1.07, sigma_c = 0.496.
Spike rule follows the COMPANION paper: y_i = sin(phi_i), spike = upward crossing of y_th = 0.6
(Buendia uses y = 1 + sin phi; companion uses y = sin phi -- we reproduce the companion).

Three mechanisms, each independently switchable for the ablation analysis (see run_prototype.py):
  1. dynamic CORRELATED a_n(t)  : a_mode='ou', a_corr in [0,1]   -> local distance-to-criticality drift
  2. common input               : g_in > 0                       -> shared afferent drive from the network
  3. between-region coupling G  : G > 0 over a SPARSE graph       -> direct inter-region connectivity

Time: internal units are simulation units; SIM_PER_SEC sim-units == 1 real second (paper's 1 sim-s
== 1 real-ms rescaling). tau_* are given in REAL seconds.
"""

from __future__ import annotations
import numpy as np
from numba import njit

SIM_PER_SEC = 1000.0
Y_TH = 0.6


def _build_csr(K, p, rng):
    """Random directed sparse connectivity (density p): receiver i <- source list. CSR format."""
    if p <= 0.0:
        return np.zeros(K + 1, np.int64), np.zeros(0, np.int64)
    indptr = np.zeros(K + 1, np.int64)
    chunks = []
    for i in range(K):
        js = np.nonzero(rng.random(K) < p)[0]
        chunks.append(js)
        indptr[i + 1] = indptr[i] + js.size
    indices = (np.concatenate(chunks) if chunks else np.zeros(0)).astype(np.int64)
    return indptr, indices


@njit(cache=True, fastmath=True)
def _run(phi0, phi1, K, n_steps, dt, sigma, omega, J, G, aa_between,
         indptr0, ind0, indptr1, ind1,
         a_is_ou, a0, a1, a_mu, a_s, a_corr, tau_a,
         g_in, in_mu, in_s, tau_in, omega_ext,
         record_every, bin_steps, n_sub, seed):
    np.random.seed(seed)
    sqdt = np.sqrt(dt)
    Fin = in_mu
    theta = 0.0
    use_in = g_in > 0.0
    ca = np.sqrt(a_corr); cb = np.sqrt(1.0 - a_corr)

    c0 = np.empty(K); s0 = np.empty(K); c1 = np.empty(K); s1 = np.empty(K)
    n_rec = n_steps // record_every + 1
    R0r = np.empty(n_rec); R1r = np.empty(n_rec)
    a0r = np.empty(n_rec); a1r = np.empty(n_rec); Fr = np.empty(n_rec)
    Zc0r = np.empty(n_rec); Zs0r = np.empty(n_rec)        # complex order parameter Z = ReZ + i ImZ
    Zc1r = np.empty(n_rec); Zs1r = np.empty(n_rec)        # (needed for the Shinomoto-Kuramoto param S)
    act0 = np.zeros(n_steps, np.int32); act1 = np.zeros(n_steps, np.int32)
    nb = n_steps // bin_steps + 1
    sc0 = np.zeros((K, nb), np.int32); sc1 = np.zeros((K, nb), np.int32)  # per-neuron binned spikes
    asub0 = np.zeros(n_steps, np.int32); asub1 = np.zeros(n_steps, np.int32)  # subsample pop activity
    yp0 = np.sin(phi0); yp1 = np.sin(phi1)
    ri = 0

    for t in range(n_steps):
        m0c = 0.0; m0s = 0.0; m1c = 0.0; m1s = 0.0
        for i in range(K):
            ci = np.cos(phi0[i]); si = np.sin(phi0[i]); c0[i] = ci; s0[i] = si; m0c += ci; m0s += si
        for i in range(K):
            ci = np.cos(phi1[i]); si = np.sin(phi1[i]); c1[i] = ci; s1[i] = si; m1c += ci; m1s += si
        m0c /= K; m0s /= K; m1c /= K; m1s /= K
        R0 = np.sqrt(m0c*m0c + m0s*m0s); R1 = np.sqrt(m1c*m1c + m1s*m1s)
        st = np.sin(theta); ct = np.cos(theta); Fc = g_in*Fin

        cnt0 = 0
        for i in range(K):
            within = J*(m0s*c0[i] - m0c*s0[i])               # J*R0*sin(Psi0 - phi0_i)
            if aa_between:
                between = G*(m1s*c0[i] - m1c*s0[i])           # G*R1*sin(Psi1 - phi0_i): all-to-all between
            else:
                accs = 0.0; accc = 0.0
                for idx in range(indptr0[i], indptr0[i+1]):
                    j = ind0[idx]; accs += s1[j]; accc += c1[j]
                between = (G/K)*(c0[i]*accs - s0[i]*accc)      # (G/K) Sum sin(phi1_j - phi0_i): sparse
            d = omega + a0*s0[i] + within + between
            if use_in:
                d += Fc*(st*c0[i] - ct*s0[i])                  # g_in*F*sin(theta - phi0_i)
            phi0[i] += d*dt + sigma*sqdt*np.random.standard_normal()
            y = np.sin(phi0[i])
            if yp0[i] < Y_TH and y >= Y_TH:
                cnt0 += 1; sc0[i, t // bin_steps] += 1
                if i < n_sub:
                    asub0[t] += 1
            yp0[i] = y

        cnt1 = 0
        for i in range(K):
            within = J*(m1s*c1[i] - m1c*s1[i])
            if aa_between:
                between = G*(m0s*c1[i] - m0c*s1[i])
            else:
                accs = 0.0; accc = 0.0
                for idx in range(indptr1[i], indptr1[i+1]):
                    j = ind1[idx]; accs += s0[j]; accc += c0[j]
                between = (G/K)*(c1[i]*accs - s1[i]*accc)
            d = omega + a1*s1[i] + within + between
            if use_in:
                d += Fc*(st*c1[i] - ct*s1[i])
            phi1[i] += d*dt + sigma*sqdt*np.random.standard_normal()
            y = np.sin(phi1[i])
            if yp1[i] < Y_TH and y >= Y_TH:
                cnt1 += 1; sc1[i, t // bin_steps] += 1
                if i < n_sub:
                    asub1[t] += 1
            yp1[i] = y
        act0[t] = cnt0; act1[t] = cnt1

        if a_is_ou:                                            # correlated OU via shared latent (corr == a_corr)
            wc = np.random.standard_normal()
            w0 = ca*wc + cb*np.random.standard_normal()
            w1 = ca*wc + cb*np.random.standard_normal()
            a0 += (a_mu - a0)/tau_a*dt + np.sqrt(2.0/tau_a)*a_s*sqdt*w0
            a1 += (a_mu - a1)/tau_a*dt + np.sqrt(2.0/tau_a)*a_s*sqdt*w1
        if use_in:
            Fin += (in_mu - Fin)/tau_in*dt + np.sqrt(2.0/tau_in)*in_s*sqdt*np.random.standard_normal()
            if Fin < 0.0:
                Fin = 0.0
        theta += omega_ext*dt

        if t % record_every == 0:
            R0r[ri] = R0; R1r[ri] = R1; a0r[ri] = a0; a1r[ri] = a1; Fr[ri] = g_in*Fin
            Zc0r[ri] = m0c; Zs0r[ri] = m0s; Zc1r[ri] = m1c; Zs1r[ri] = m1s; ri += 1

    return (R0r[:ri], R1r[:ri], a0r[:ri], a1r[:ri], Fr[:ri], act0, act1, sc0, sc1, asub0, asub1,
            Zc0r[:ri], Zs0r[:ri], Zc1r[:ri], Zs1r[:ri])


def simulate(K=250, T=2000.0, dt=0.01, sigma=0.499, omega=1.0, J=1.0,
             G=0.1, p_between=0.2, aa_between=False,
             a_mode='static', a_value=1.07, a_value1=None,
             a_mu=1.07, a_s=0.08, a_corr=0.7, tau_a_sec=5.0,
             g_in=0.0, in_fluct=0.3, tau_in_sec=5.0, omega_ext=1.0,
             seed=0, record_every=50, spike_bin_sec=0.02, n_sub=0):
    """Run the two-population model.

    G, p_between : strength and DENSITY of the sparse between-region graph (within is all-to-all).
                   Effective inter-region coupling ~ G * p_between. Set G=0 to ablate coupling.
    aa_between   : if True, neurons between regions are fully connected and p_between is ignored.
    a_mode       : 'static' (a fixed) or 'ou' (correlated slow OU per hemisphere).
    a_corr       : target correlation between a0(t) and a1(t) in [0,1] (0=indep, 1=identical).
    g_in         : GLOBAL scale of the common input field F(t)*sin(theta-phi). Set 0 to ablate.
    tau_*_sec    : OU correlation times in real seconds.
    """
    rng = np.random.default_rng(seed)
    phi0 = rng.uniform(0, 2*np.pi, K).astype(np.float64)
    phi1 = rng.uniform(0, 2*np.pi, K).astype(np.float64)
    indptr0, ind0 = _build_csr(K, p_between, rng)   # pop0 receives from pop1
    indptr1, ind1 = _build_csr(K, p_between, rng)   # pop1 receives from pop0
    n_steps = int(round(T/dt))
    if a_mode == 'static':
        a0 = a_value
        a1 = a_value if a_value1 is None else a_value1     # asymmetric static a per population
    else:
        a0 = a1 = a_mu
    bin_steps = max(1, int(round(spike_bin_sec*SIM_PER_SEC/dt)))

    (R0, R1, a0r, a1r, F, act0, act1, sc0, sc1, asub0, asub1,
     ReZ0, ImZ0, ReZ1, ImZ1) = _run(
        phi0, phi1, K, n_steps, dt, sigma, omega, J, G, bool(aa_between),
        indptr0, ind0, indptr1, ind1,
        a_mode == 'ou', a0, a1, a_mu, a_s, float(np.clip(a_corr, 0, 1)), tau_a_sec*SIM_PER_SEC,
        g_in, 1.0, in_fluct, tau_in_sec*SIM_PER_SEC, omega_ext,
        record_every, bin_steps, int(n_sub), seed)

    t_rec = np.arange(R0.size)*record_every*dt
    return dict(R0=R0, R1=R1, a0=a0r, a1=a1r, F=F, t_rec=t_rec,
                act0=act0, act1=act1, sc0=sc0, sc1=sc1, asub0=asub0, asub1=asub1,
                ReZ0=ReZ0, ImZ0=ImZ0, ReZ1=ReZ1, ImZ1=ImZ1,
                dt=dt, n_steps=n_steps, K=K, G=G,
                p_between=p_between, a_mode=a_mode, a_corr=a_corr, g_in=g_in,
                tau_a_sec=tau_a_sec, record_every=record_every, spike_bin_sec=spike_bin_sec)
