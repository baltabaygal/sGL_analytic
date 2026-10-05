#!/usr/bin/env python
"""Semi-analytic sGL magnification PDF -- halos-only, spherical, unbiased Poisson.

Chain (no Monte Carlo, no fitting, no simulation input anywhere):

    dn/dlnM  ->  R(xi)  ->  Lambda(k)  ->  P(xi)  ->  dP/dmu

Physics is the validated chain of ``scripts/comparisons/analytic_pdf.py`` /
``analytic_chain_spec.md`` (every spec checkpoint is re-verified by
``checkpoints.py`` in this folder). What is NEW here relative to that script is
the last two links -- the Levy-Khintchine exponent ``Lambda_of_k`` and the
Fourier inversion ``P_of_xi`` -- which the spec describes (its Sec. 7) but the
reference script never implemented.

Model scope: spherical NFW lenses with the exact Wright-Brainerd projection,
Sheth-Tormen mass function, Poisson lens counts, NO linear bias, NO filaments,
NO ellipticity, NO subhalos.  That is deliberately the scope of the C++ engine
run with ``filaments=False, bias=False, ell=False, subhalo=False``.

Two conventions are selectable so the chain can be run either standalone or
matched to the C++ engine:

  ``window="tophat"``  real-space top-hat sigma(R) normalised to sigma8
                       (the spec / Vaskonen paper Sec. 2 convention)
  ``window="smoothk"`` the engine's smooth-k filter Ws(x)=1/(1+(0.43x)^6),
                       with sigma8 imposed *through that same filter* at
                       R = 8/h Mpc -- i.e. what ``cpp/cosmology.cpp::sigmaC``
                       actually does.  Reproduces the engine's sigma(M).

  ``transfer="nowiggle"``  Eisenstein-Hu 1998 no-wiggle (spec convention)
  ``transfer="eh98"``      full EH98 with the acoustic oscillations, which is
                           what ``cpp/cosmology.cpp::TM`` uses.

Units: Msun, Mpc (NOT Mpc/h), km/s/Mpc.
"""
from __future__ import annotations

import numpy as np
from numpy import log, exp, sqrt, sin, cos, pi
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import gamma as Gamma

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

try:
    import numba as _numba
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

# ============================== 0. constants ================================
# Planck-2018 benchmark == the C++ engine defaults.
DEFAULTS = dict(h=0.674, Om=0.315, Ob=0.0493, s8=0.811, ns=0.965,
                TCMB=2.7255, dc=1.686, zeq=3402.0)

CKMS = 299792.458            # km/s
RHOC0_UNIT = 2.77536627e11   # Msun/Mpc^3 / h^2
GNEWT = 4.30091e-9           # Mpc (km/s)^2 / Msun


class Cosmology:
    """Background + linear-theory sector.  Everything downstream reads this."""

    def __init__(self, h=None, Om=None, Ob=None, s8=None, ns=None,
                 TCMB=None, dc=None, zeq=None,
                 window="tophat", transfer="nowiggle", conc_model=14,
                 anchor="tophat"):
        d = DEFAULTS
        self.h = d["h"] if h is None else h
        self.Om = d["Om"] if Om is None else Om
        self.Ob = d["Ob"] if Ob is None else Ob
        self.s8 = d["s8"] if s8 is None else s8
        self.ns = d["ns"] if ns is None else ns
        self.TCMB = d["TCMB"] if TCMB is None else TCMB
        if anchor not in ("tophat", "window"):
            raise ValueError('anchor must be "tophat" or "window"')
        self.anchor = anchor
        if conc_model not in (14, 16):
            raise ValueError("conc_model must be 14 or 16")
        self.conc_model = conc_model
        self.dc = d["dc"] if dc is None else dc
        self.zeq = d["zeq"] if zeq is None else zeq
        self.OL = 1.0 - self.Om
        self.H0 = 100.0 * self.h
        self.rhoc0 = RHOC0_UNIT * self.h**2
        self.rhom = self.Om * self.rhoc0        # comoving matter density
        if window not in ("tophat", "smoothk"):
            raise ValueError("window must be 'tophat' or 'smoothk'")
        if transfer not in ("nowiggle", "eh98"):
            raise ValueError("transfer must be 'nowiggle' or 'eh98'")
        self.window = window
        self.transfer = transfer
        self._build_growth()
        self._build_power()
        self._build_sigma()

    # ---------------------------- background --------------------------------
    def E(self, z):
        return sqrt(self.Om * (1 + z)**3 + self.OL)

    def Hz(self, z):
        return self.H0 * self.E(z)

    def rhocrit(self, z):                       # physical
        return self.rhoc0 * self.E(z)**2

    def chi(self, z):                           # comoving distance [Mpc]
        """MEMOISED per exact z (2026-09-25).  The chain only ever asks for the
        ~103 engine-grid redshifts and z_s, but Sigma_cr calls chi 3x per call
        and nfw_params calls Sigma_cr 184k times per full-config build: 738k
        identical quads (~12 s).  Returning the cached quad result is
        bit-identical.  (A spline was tried and REJECTED: this quad is already
        accurate to 4e-15, the spline only to 2e-10 -- unlike D(z), where the
        per-call quad was the inaccurate side.)"""
        key = float(z)
        c = self.__dict__.setdefault("_chi_cache", {})
        v = c.get(key)
        if v is None:
            v = quad(lambda zz: CKMS / self.H0 / self.E(zz), 0, z)[0]
            c[key] = v
        return v

    def DA(self, z):
        return self.chi(z) / (1 + z)

    def DA_ls(self, zl, zs):                    # flat universe
        return (self.chi(zs) - self.chi(zl)) / (1 + zs)

    def Sigma_cr(self, zl, zs):                 # Msun/Mpc^2, physical D_A's
        return (CKMS**2 * self.DA(zs)
                / (4 * pi * GNEWT * self.DA(zl) * self.DA_ls(zl, zs)))

    # ------------------------------ growth ----------------------------------
    def _build_growth(self):
        """Unnormalised growth D_un(a) = E(a) * I(a),
        I(a) = int_0^a da' / (a' E(a'))^3.

        I is TABULATED ONCE and splined (2026-09-25).  It used to be a fresh
        adaptive `quad` on every single call, which a cProfile of one
        `build_population(+sub)` at z_s=1 caught as 913,705 `quad` calls
        driving 46.3M integrand evaluations -- ~40 s of an 893 s build, for a
        smooth function of ONE variable.

        The spline is in log-log because the integrand is a clean power law at
        small a: E(a) -> sqrt(Om) a^{-3/2}, so (a E)^3 -> Om^{3/2} a^{-3/2}
        and I(a) -> (2/5) a^{5/2} / Om^{3/2}.  Splining log I against log a
        therefore interpolates something very close to a straight line.
        E(a) itself stays analytic and is NOT splined, so all of the fast
        a-dependence is carried exactly.
        """
        def Ea(a):
            return sqrt(self.Om * np.asarray(a, float)**-3 + self.OL)

        # a in [1e-6, 1] covers z in [0, 1e6]; the chain never goes above the
        # engine's z_max = 12.34 (a = 0.0749), so this is far outside any use.
        ag = np.geomspace(1e-6, 1.0, 4096)
        seg = np.empty(ag.size)
        seg[0] = quad(lambda ap: 1.0 / (ap * Ea(ap))**3, 0.0, ag[0])[0]
        for i in range(1, ag.size):
            seg[i] = quad(lambda ap: 1.0 / (ap * Ea(ap))**3,
                          ag[i - 1], ag[i])[0]
        Ig = np.cumsum(seg)
        self._lnI = CubicSpline(log(ag), log(Ig))
        self._a_growth_range = (float(ag[0]), float(ag[-1]))

        def Dun(a):
            a = np.asarray(a, float)
            return Ea(a) * exp(self._lnI(log(a)))

        self._Ea = Ea
        self._Dun = Dun
        self._D0 = float(Dun(1.0))

    def D(self, z):
        a = 1.0 / (1 + np.asarray(z, float))
        out = self._Dun(a) / self._D0
        return float(out) if np.isscalar(z) or np.ndim(z) == 0 else out

    # ------------------------ transfer function -----------------------------
    def _T_nowiggle(self, k):
        """Eisenstein & Hu 1998 no-wiggle (spec Sec. 1)."""
        wm, wb = self.Om * self.h**2, self.Ob * self.h**2
        th = self.TCMB / 2.7
        s = 44.5 * log(9.83 / wm) / sqrt(1 + 10 * wb**0.75)
        aG = (1 - 0.328 * log(431 * wm) * (wb / wm)
              + 0.38 * log(22.3 * wm) * (wb / wm)**2)
        Geff = self.Om * self.h * (aG + (1 - aG) / (1 + (0.43 * k * s)**4))
        q = k * th * th / (Geff * self.h)
        L = log(2 * np.e + 1.8 * q)
        C = 14.2 + 731.0 / (1 + 62.5 * q)
        return L / (L + C * q * q)

    def _T_eh98(self, k):
        """Full Eisenstein & Hu 1998 CDM+baryon transfer function.

        Transcribed to match ``cpp/cosmology.cpp::TM`` term for term (that
        routine is itself EH98 with k in Mpc^-1 and the sound horizon in Mpc).
        """
        h, Om, Ob = self.h, self.Om, self.Ob
        Oc = Om - Ob
        wm, wb = Om * h**2, Ob * h**2
        T0 = self.TCMB
        # z_eq is a free input in the engine (default 3402) and it sets
        # OmegaR = OmegaM/(1+zeq); with radiation included H(zeq)^2 =
        # 2 OmegaM H0^2 (1+zeq)^3, so k_eq = a_eq H(a_eq)/c reduces to
        zeq = self.zeq
        keq = (self.H0 / CKMS) * sqrt(2.0 * Om) * sqrt(1.0 + zeq)   # Mpc^-1
        ksilk = (0.0016 * wb**0.52 * wm**0.73
                 * (1 + (10.4 * wm)**-0.95))
        a1 = (46.9 * wm)**0.67 * (1 + (32.1 * wm)**-0.532)
        a2 = (12.0 * wm)**0.424 * (1 + (45.0 * wm)**-0.582)
        b1 = 0.944 / (1 + (458 * wm)**-0.708)
        b2 = (0.395 * wm)**-0.026
        ac = a1**(-Ob / Om) * a2**(-(Ob / Om)**3)
        bc = 1.0 / (1 + b1 * ((Oc / Om)**b2 - 1))
        s = 44.5 * log(9.83 / wm) / sqrt(1 + 10 * wb**0.75)     # Mpc
        b3 = 0.313 * wm**-0.419 * (1 + 0.607 * wm**0.674)
        b4 = 0.238 * wm**0.223
        zd = (1291 * wm**0.251 / (1 + 0.659 * wm**0.828) * (1 + b3 * wb**b4))
        Rd = 31.5 * wb * (T0 / 2.7)**-4 / (zd / 1000.0)

        def g2(y):
            return y * (-6 * sqrt(1 + y)
                        + (2.0 + 3.0 * y) * log((sqrt(1 + y) + 1)
                                                / (sqrt(1 + y) - 1)))

        ab = 2.07 * keq * s * (1 + Rd)**-0.75 * g2((1 + zeq) / (1 + zd))
        bb = (0.5 + Ob / Om
              + (3.0 - 2.0 * Ob / Om) * sqrt(1 + (17.2 * wm)**2))
        bnode = 8.41 * wm**0.435

        q = k / (13.41 * keq)
        fk = 1.0 / (1 + (k * s / 5.4)**4)

        def To1(kk, a_c, b_c):
            qq = kk / (13.41 * keq)
            C1 = 14.2 / a_c + 386.0 / (1 + 69.9 * qq**1.08)
            num = log(np.e + 1.8 * b_c * qq)
            return num / (num + C1 * qq * qq)

        TC = fk * To1(k, 1.0, bc) + (1 - fk) * To1(k, ac, bc)
        s3 = s / (1 + (bnode / (k * s))**3)**(1.0 / 3.0)
        x = k * s3
        jo = np.where(x < 1e-8, 1.0 - x * x / 6.0, sin(x) / np.where(x == 0, 1, x))
        TB = ((To1(k, 1.0, 1.0) / (1 + (k * s / 5.2)**2)
               + ab / (1 + (bb / (k * s))**3) * exp(-(k / ksilk)**1.4)) * jo)
        return Ob / Om * TB + Oc / Om * TC

    def T(self, k):
        return (self._T_nowiggle(k) if self.transfer == "nowiggle"
                else self._T_eh98(k))

    # --------------------------- sigma(M) -----------------------------------
    def _W(self, x):                             # real-space top hat
        return 3 * (sin(x) - x * cos(x)) / x**3

    def _Ws(self, x):                            # engine smooth-k filter
        return 1.0 / (1.0 + (0.43 * x)**6)

    def _build_power(self):
        self._kg = np.logspace(-5, 3, 4000)
        self._Pshape = self._kg**self.ns * self.T(self._kg)**2

    def _sigmaR_shape(self, R, window=None):
        w = self.window if window is None else window
        x = self._kg * R
        Wk = self._W(x) if w == "tophat" else self._Ws(x)
        return sqrt(_trapz(self._kg**2 * self._Pshape * Wk**2, self._kg)
                    / (2 * pi * pi))

    def _build_sigma(self):
        # sigma8 ANCHOR (2026-09-09). Until the engine's `sigma8_tophat` toggle
        # was removed it anchored through the SELECTED window, so smooth-k mode
        # anchoring through Ws reproduced it and an input 0.811 meant a top-hat
        # sigma8 of 0.7786. The engine now anchors through the REAL-SPACE TOP-HAT
        # unconditionally (deltaH8 = sigma8/sigmaTH(M8,1)) while still using Ws
        # for sigma_M(M), so anchoring through Ws here leaves the chain ~4.2% low
        # in sigma and ~8.5% low in P(k) against the engine -- first-order, and
        # comparable to the whole legacy->paper composition shift.
        #
        # `anchor` therefore defaults to "tophat", which is a NO-OP for
        # window="tophat" (the spec/paper convention) and RESTORES the engine
        # match for window="smoothk". Pass anchor="window" to reproduce
        # pre-2026-09-09 smooth-k results exactly.
        self._Anorm = self.s8 / self._sigmaR_shape(
            8.0 / self.h, window=("tophat" if self.anchor == "tophat" else None))
        Mg = np.logspace(4, 17, 500)
        lnsig = np.array([log(self.sigmaR(self.RofM(M))) for M in Mg])
        self._spl = CubicSpline(log(Mg), lnsig)

    def sigmaR(self, R):
        return self._Anorm * self._sigmaR_shape(R)

    def sigma_tophat8(self):
        """sigma8 measured with a real-space top hat, whatever the filter used
        for the normalisation.  Engine value: 0.7786."""
        return self._Anorm * self._sigmaR_shape(8.0 / self.h, window="tophat")

    def RofM(self, M):
        return (3 * M / (4 * pi * self.rhom))**(1.0 / 3.0)

    def sigmaM(self, M, z=0.0):
        return exp(self._spl(log(M))) * self.D(z)

    def dlnsig_dlnM(self, M):
        return self._spl(log(M), 1)

    # ----------------------- Sheth-Tormen mass function ---------------------
    P_ST, Q_ST = 0.3, 0.8
    A_ST = 1.0 / (1 + 2**-0.3 * Gamma(0.5 - 0.3) / sqrt(pi))

    def dndlnM(self, M, z):
        """Comoving number density per lnM [Mpc^-3].  nu = delta_c^2/sigma^2."""
        nu = (self.dc / self.sigmaM(M, z))**2
        qn = self.Q_ST * nu
        mult = (self.A_ST * (1 + qn**-self.P_ST) * sqrt(qn / (2 * pi))
                * exp(-qn / 2))
        return (self.rhom / M) * mult * (-2 * self.dlnsig_dlnM(M))

    # ------------------------------ NFW -------------------------------------
    def conc(self, M, z, h):
        """NFW 200c concentration; ``self.conc_model`` selects the relation.

        14 -- Dutton & Maccio 2014 (== the engine's ``cons14``, the default).
        16 -- 1601.02624 (== the engine's ``cons16`` AFTER the supervisor's
              nu0 /= Dg(z) fix, halos 991950e).

        The two differ in kind, not just in coefficients: cons14 is a fit in
        (M, z) with NO power-spectrum dependence, so at fixed (M, z, h) it is
        identically flat in sigma8; cons16 is a fit in peak height
        nu = delta_c(z)/sigma(M), so cosmology enters through sigma(M).
        Measured in the engine: sigma8 0.60 -> 1.20 moves cons14 by 1.0000x at
        every mass and cons16 by ~1.42-1.52x.

        Growth CANCELS in cons16: the engine has nu = deltac0/(Dg sigma_0) and
        nu0 = poly(z)/Dg, so nu/nu0 = deltac0/(sigma_0 poly(z)) with sigma_0
        the z = 0 linear sigma. That is why sigmaM(M, 0.0) appears below and no
        D(z) does -- do not "restore" a growth factor here.
        """
        if getattr(self, "conc_model", 14) == 16:
            a = 1.0 + z
            poly = (4.135 - 0.564 * a - 0.210 * a**2
                    + 0.0557 * a**3 - 0.00348 * a**4)
            # cons16's nu0 quartic crosses zero at z = 7.7975; the engine clamps
            # its z at 7 (cosmology.cpp:415) rather than continue past it, and
            # this mirrors that clamp exactly so the two agree above z = 7.
            if z > 7.0:
                return self.conc(M, 7.0, h)
            q = self.dc / (self.sigmaM(M, 0.0) * poly)
            c0 = 3.395 * a**-0.215
            beta = 0.307 * a**0.540
            g1 = 0.628 * a**-0.047
            g2 = 0.317 * a**-0.893
            return c0 * q**-g1 * (1.0 + q**(1.0 / beta))**(-beta * (g2 - g1))
        a = 0.520 + (0.905 - 0.520) * exp(-0.617 * z**1.21)
        b = -0.101 + 0.026 * z
        return 10.0**(a + b * np.log10(M * h / 1e12))

    def nfw_params(self, M, zl, zs):
        C = self.conc(M, zl, self.h)
        r200 = (3 * M / (4 * pi * 200 * self.rhocrit(zl)))**(1.0 / 3.0)
        rs = r200 / C
        fC = log(1 + C) - C / (1 + C)
        rhos = (200.0 / 3.0) * self.rhocrit(zl) * C**3 / fC
        ks = rhos * rs / self.Sigma_cr(zl, zs)
        return C, rs, ks, fC


# ===================== Wright & Brainerd projected NFW =======================
# kappa = 2 ks F(x), kbar = 4 ks h(x)/x^2, gamma = kbar - kappa, with
#   F = [1 - A/s] / (x^2 - 1),  h = ln(x/2) + A/s,   w = 1 - x^2,  s = sqrt|w|,
#   A = arccosh(1/x) (x < 1),  arccos(1/x) = arctan(s) (x > 1).
# Precision (2026-10-03).  A/s = atanh(s)/s or atan(s)/s = sum_n w^n/(2n+1) on
# BOTH sides of x = 1, so for |w| < NFW_W_SERIES F = sum_n w^n/(2n+3) and
# h = ln(x/2) + sum_n w^n/(2n+1) (NFW_N_SERIES terms, truncation < 1e-18).
# Outside it: w = (1-x)(1+x) and arccosh(1/x) = log1p(e + sqrt(e(e+2))),
# e = (1-x)/x, exact; for x < NFW_X_H the cancellation ln(x/2) + A/s is removed
# algebraically (1 - s = x^2/(1+s)):
#   h = [ln(2/x) x^2/(1+s) + log1p(-x^2/(2(1+s)))] / s.
# Replaces a +-1e-6 linear band around x = 1 (whose h slope was wrong: -1/3
# instead of -2/3 -> kbar off by 1.1e-6 at x = 1 +- 1e-6), the 1 - x*x
# cancellation just outside it (kappa off by up to 8e-6 at x = 1 + 2e-6) and
# the 2-term small-x series of h below x = 1e-3 (kbar off by 7e-7 at 1e-3).
# Checked vs 40-digit quadrature of the 3D density (303 points, x = 1e-6 .. 1e3
# and x = 1 +- 1e-7 .. 5e-2): kappa <= 8.9e-16, kbar <= 5.6e-16 rel (was 8.3e-6 /
# 1.1e-6); vs the 80-digit closed form on 6223 points (1e-8 .. 1e4, dense at
# every branch switch): kappa <= 2.4e-15, kbar <= 8.9e-16 (was 3.5e-5 / 1.1e-6).
# gamma = kbar - kappa keeps the inherent small-x cancellation (<= 1.1e-14).
NFW_W_SERIES = 0.25
NFW_N_SERIES = 28          # 0.25^28 / 59 = 2.4e-19
NFW_X_H = 0.9


def _nfw_Fh_numpy(x):
    x = np.asarray(x, float)
    F = np.empty_like(x)
    h = np.empty_like(x)
    w = (1.0 - x) * (1.0 + x)
    ser = np.abs(w) < NFW_W_SERIES
    lo = ~ser & (x < 1.0)
    hi = ~ser & (x > 1.0)
    ws = w[ser]
    T = np.zeros_like(ws)
    for n in range(NFW_N_SERIES - 1, -1, -1):           # T = sum w^n/(2n+3)
        T = T * ws + 1.0 / (2 * n + 3)
    F[ser] = T
    h[ser] = log(0.5 * x[ser]) + (1.0 + ws * T)
    xl, wl = x[lo], w[lo]
    e = (1.0 - xl) / xl
    s = sqrt(wl)
    A = np.log1p(e + sqrt(e * (e + 2.0)))
    F[lo] = (A / s - 1.0) / wl
    hs = np.where(xl < NFW_X_H,
                  (log(2.0 / xl) * xl * xl / (1.0 + s)
                   + np.log1p(-xl * xl / (2.0 * (1.0 + s)))) / s,
                  log(0.5 * xl) + A / s)
    h[lo] = hs
    xh = x[hi]
    v = -w[hi]
    sv = sqrt(v)
    B = np.arctan(sv)
    F[hi] = (1.0 - B / sv) / v
    h[hi] = log(0.5 * xh) + B / sv
    return F, h


def _F_numpy(x):
    return _nfw_Fh_numpy(x)[0]


def _hfun_numpy(x):
    return _nfw_Fh_numpy(x)[1]


def _kappa_gamma_numpy(x, ks):
    F, h = _nfw_Fh_numpy(x)
    k = 2 * ks * F
    kbar = 4 * ks * h / np.asarray(x, float)**2
    return k, kbar - k


if _HAVE_NUMBA:
    import math as _math

    @_numba.njit(cache=True)
    def nfw_Fh(x):
        """(F(x), h(x)) of the projected NFW profile at one point; see the
        comment above NFW_W_SERIES.  Shared by kappa_gamma and the numba
        kernels of subhalos_exact.  No fastmath: the exact forms of w, e and
        the cancellation-free h must not be re-associated."""
        w = (1.0 - x) * (1.0 + x)
        if abs(w) < NFW_W_SERIES:
            T = 0.0
            for n in range(NFW_N_SERIES - 1, -1, -1):
                T = T * w + 1.0 / (2 * n + 3)
            return T, _math.log(0.5 * x) + (1.0 + w * T)
        if x < 1.0:
            e = (1.0 - x) / x
            s = _math.sqrt(w)
            A = _math.log1p(e + _math.sqrt(e * (e + 2.0)))
            F = (A / s - 1.0) / w
            if x < NFW_X_H:
                h = (_math.log(2.0 / x) * x * x / (1.0 + s)
                     + _math.log1p(-x * x / (2.0 * (1.0 + s)))) / s
            else:
                h = _math.log(0.5 * x) + A / s
            return F, h
        v = -w
        sv = _math.sqrt(v)
        B = _math.atan(sv)
        return (1.0 - B / sv) / v, _math.log(0.5 * x) + B / sv

    @_numba.njit(cache=True)
    def _kappa_gamma_numba(x, ks):
        n = x.shape[0]
        kap = np.empty(n, dtype=np.float64)
        gam = np.empty(n, dtype=np.float64)
        for i in range(n):
            xi = x[i]
            F, h = nfw_Fh(xi)
            k = 2.0 * ks * F
            kbar = 4.0 * ks * h / (xi * xi)
            kap[i] = k
            gam[i] = kbar - k
        return kap, gam

    def kappa_gamma(x, ks):
        x = np.ascontiguousarray(x, dtype=np.float64)
        return _kappa_gamma_numba(x, float(ks))
else:
    kappa_gamma = _kappa_gamma_numpy


# ======================= jump-measure grid and binning =======================
# XI reaches 30: the near-caustic exponential floor of the cluster cross-section
# lives at xi ~ 1-30 and the spec's original cap of 4 truncated it.
XI = np.logspace(-7, np.log10(30.0), 560)
_lnXI = log(XI)
_XIE = np.empty(XI.size + 1)
_XIE[1:-1] = sqrt(XI[1:] * XI[:-1])
_XIE[0] = XI[0]**2 / _XIE[1]
_XIE[-1] = XI[-1]**2 / _XIE[-2]
_XIW = np.diff(_XIE)

_XG = np.logspace(-6, 3.5, 900)


_BISECT_ITERS = 45   # 2^-45 = 2.8e-14 relative; eps grid below only resolves
                     # down to 1e-11, so this is still >>enough margin (was 70)


def _refined_xgrid(ks):
    """Base log grid plus adaptive refinement around every zero of detA(x).

    Without this the thin annulus at the tangential critical curve of a
    cluster-scale lens is unresolved and dsigma/dxi is under-counted by up to
    ~4x for xi >~ 0.5 (spec Sec. 9 warning; the erratum found 2026-07-24).

    2026-09-04 perf: all crossings of one lens are bisected TOGETHER in a
    single vectorized kappa_gamma call per iteration, instead of one Python
    scalar call per iteration per crossing (was up to 70 scalar calls *per
    crossing*, profiled at 86k calls total for a single R_of_xi). Same exact
    bisection algorithm and root, just batched -- verified bitwise-identical
    xc for a fixed number of shared iterations (see checkpoints.py diff).
    """
    k, g = kappa_gamma(_XG, ks)
    detA = (1 - k)**2 - g * g
    sgn = np.sign(detA)
    idx = np.nonzero(sgn[1:] * sgn[:-1] < 0)[0]
    xs = [_XG]
    if idx.size:
        a = _XG[idx].copy()
        b = _XG[idx + 1].copy()
        sgn0 = sgn[idx]
        for _ in range(_BISECT_ITERS):             # bisect detA = 0
            m = 0.5 * (a + b)
            km, gm = kappa_gamma(m, ks)
            same = np.sign((1 - km)**2 - gm * gm) == sgn0
            a = np.where(same, m, a)
            b = np.where(same, b, m)
        xc = 0.5 * (a + b)
        eps = np.logspace(-11, -1.2, 22)
        xs.append((xc[:, None] * (1.0 + eps[None, :])).ravel())
        xs.append((xc[:, None] * (1.0 - eps[None, :])).ravel())
    x = np.unique(np.concatenate(xs))
    return x[x > 0]


def dsigma_bins(cos, M, zl, zs):
    """Deposit-binned single-lens cross-section dsigma/dxi on XI [Mpc^2].

    Every x-interval deposits its exact annulus area pi rs^2 (x2^2 - x1^2)
    uniformly over the xi range it spans -- branch-safe and area-exact, unlike
    the |dxi/dlnx| Jacobian recipe, which merges branches at the caustic.
    """
    C, rs, ks, fC = cos.nfw_params(M, zl, zs)
    x = _refined_xgrid(ks)
    k, g = kappa_gamma(x, ks)
    detA = (1 - k)**2 - g * g
    good = detA > 1e-300
    xi = np.where(good, -log(np.where(good, detA, 1.0)), np.nan)
    xa, xb = x[:-1], x[1:]
    fa, fb = xi[:-1], xi[1:]
    ok = np.isfinite(fa) & np.isfinite(fb)
    xa, xb, fa, fb = xa[ok], xb[ok], fa[ok], fb[ok]
    lo, hi = np.minimum(fa, fb), np.maximum(fa, fb)
    dsig = pi * rs * rs * (xb * xb - xa * xa)
    keep = (hi > _XIE[0]) & (lo < _XIE[-1]) & (dsig > 0)
    if not np.any(keep):
        return np.zeros_like(XI)
    lo, hi, dsig = lo[keep], hi[keep], dsig[keep]
    den = np.maximum(hi - lo, 1e-300)
    frac = np.clip((_XIE[:, None] - lo[None, :]) / den[None, :], 0.0, 1.0)
    return np.diff(frac @ dsig) / _XIW


def campbell_moments(cos, zs, kappa_min, Mmin=1e7, Mmax=1e16, Nz=40, NM=48):
    """Campbell (compound-Poisson) moments of the ADDITIVE fields kappa and
    gamma, integrated over the same lens population as ``R_of_xi``.

    For a compensated Poisson sum, Var(kappa) = int kappa^2 dR exactly and
    <|sum gamma_i|^2> = int gamma^2 dR (cross terms vanish for random lens
    position angles).  These involve NO scalar reduction and no Fourier
    inversion -- they test abundance x profile x geometry x counts alone, so
    they separate an error in R from an error in the kappa -> mu composition.

    Each halo is integrated out to the radius where its own convergence falls
    to ``kappa_min``, matching the engine's coverage: explicit lenses above
    kappa_thr plus the Gaussian background down to eps_floor * kappa_thr.
    """
    zgrid = np.linspace(1e-3, zs - 1e-3, Nz)
    dz = zgrid[1] - zgrid[0]
    Mgrid = np.logspace(np.log10(Mmin), np.log10(Mmax), NM)
    dlnM = log(Mgrid[1]) - log(Mgrid[0])
    acc = dict(kappa=0.0, kappa2=0.0, gamma2=0.0, xi=0.0, xi2=0.0, N=0.0)
    for z in zgrid:
        wz = (1 + z)**2 * CKMS / cos.Hz(z) * dz
        for M in Mgrid:
            C, rs, ks, fC = cos.nfw_params(M, z, zs)
            k, g = kappa_gamma(_XG, ks)
            sel = k > kappa_min                    # engine's radial coverage
            if not np.any(sel):
                continue
            x, kk, gg = _XG[sel], k[sel], g[sel]
            detA = (1 - kk)**2 - gg * gg
            xi = np.where(detA > 0, -log(np.where(detA > 0, detA, 1.0)), 0.0)
            pref = wz * cos.dndlnM(M, z) * dlnM * 2 * pi * rs * rs
            acc["N"] += pref * _trapz(x, x)
            acc["kappa"] += pref * _trapz(x * kk, x)
            acc["kappa2"] += pref * _trapz(x * kk * kk, x)
            acc["gamma2"] += pref * _trapz(x * gg * gg, x)
            acc["xi"] += pref * _trapz(x * xi, x)
            acc["xi2"] += pref * _trapz(x * xi * xi, x)
    return acc


def _cosmology_kwargs(cos):
    """Plain-kwarg snapshot of a Cosmology -- Cosmology itself is NOT
    picklable (``_Dun`` is a closure built in ``_build_growth``), so
    multiprocess workers rebuild it from this instead of pickling ``cos``."""
    return dict(h=cos.h, Om=cos.Om, Ob=cos.Ob, s8=cos.s8, ns=cos.ns,
                TCMB=cos.TCMB, dc=cos.dc, zeq=cos.zeq,
                window=cos.window, transfer=cos.transfer,
                conc_model=cos.conc_model, anchor=cos.anchor)


def _R_of_xi_zchunk(payload):
    """Worker: partial R(xi) summed over one chunk of the z-grid. Rebuilds
    Cosmology locally (~0.04 s) rather than pickling it."""
    cos_kwargs, zchunk, dz, Mmin, Mmax, NM, zs = payload
    cos = Cosmology(**cos_kwargs)
    Mgrid = np.logspace(np.log10(Mmin), np.log10(Mmax), NM)
    dlnM = log(Mgrid[1]) - log(Mgrid[0])
    R = np.zeros_like(XI)
    for z in zchunk:
        wz = (1 + z)**2 * CKMS / cos.Hz(z) * dz
        for M in Mgrid:
            R += wz * cos.dndlnM(M, z) * dlnM * dsigma_bins(cos, M, z, zs)
    return R


def R_of_xi(cos, zs, Mmin=1e7, Mmax=1e16, Nz=40, NM=48, nproc=1):
    """Line-of-sight-integrated jump measure R(xi; zs) [per lnmu, per l.o.s.].

    R = int dz (1+z)^2 c/H(z) int dlnM dn/dlnM dsigma/dxi   (spec Sec. 5)

    ``nproc>1`` splits the z-grid across a process pool (embarrassingly
    parallel -- every (z,M) term is independent, same pattern as the C++
    engine's process-level ray sharding). Default nproc=1 keeps the
    single-process behaviour and exact floating-point sum order used by
    every caller/checkpoint to date; results with nproc>1 agree to float
    noise (different summation order across chunks), not bitwise.
    """
    zgrid = np.linspace(1e-3, zs - 1e-3, Nz)
    dz = zgrid[1] - zgrid[0]
    if nproc is None or nproc <= 1:
        Mgrid = np.logspace(np.log10(Mmin), np.log10(Mmax), NM)
        dlnM = log(Mgrid[1]) - log(Mgrid[0])
        R = np.zeros_like(XI)
        for z in zgrid:
            wz = (1 + z)**2 * CKMS / cos.Hz(z) * dz
            for M in Mgrid:
                R += wz * cos.dndlnM(M, z) * dlnM * dsigma_bins(cos, M, z, zs)
        return R
    import concurrent.futures as _cf
    cos_kwargs = _cosmology_kwargs(cos)
    chunks = [c for c in np.array_split(zgrid, nproc) if c.size]
    payload = [(cos_kwargs, c, dz, Mmin, Mmax, NM, zs) for c in chunks]
    with _cf.ProcessPoolExecutor(max_workers=min(nproc, len(payload))) as ex:
        parts = list(ex.map(_R_of_xi_zchunk, payload))
    return np.sum(parts, axis=0)


# ==================== Levy-Khintchine exponent and inversion =================
def tilt_source(R):
    """Source-plane Esscher tilt R_s(xi) = e^{-xi} R(xi) (spec Sec. 6)."""
    return exp(-XI) * R


def _fine_grid(R, nfine=6000):
    """Log-refine R onto a denser xi grid for the oscillatory k-integral.

    R is smooth in log-log, so linear interpolation of ln R vs ln xi is
    lossless here; the refinement only buys resolution of e^{i k xi}.
    """
    xif = np.logspace(log(XI[0]) / log(10), log(XI[-1]) / log(10), nfine)
    pos = R > 0
    lnR = np.interp(log(xif), log(XI[pos]), log(R[pos]),
                    left=-np.inf, right=-np.inf)
    return xif, np.where(np.isfinite(lnR), exp(lnR), 0.0)


def _lambda_k_chunk(payload):
    """Worker: Lambda(k) on one slice of the k array, chunked internally the
    same way the serial path does (bounds peak memory per process)."""
    kc, xif, Rf, chunk = payload
    out = np.empty(kc.size, complex)
    for i0 in range(0, kc.size, chunk):
        kk = kc[i0:i0 + chunk][:, None]
        ph = kk * xif[None, :]
        integ = Rf[None, :] * (np.exp(1j * ph) - 1.0 - 1j * ph)
        out[i0:i0 + chunk] = _trapz(integ, xif, axis=-1)
    return out


def Lambda_of_k(k, R, nfine=6000, chunk=4000, nproc=1):
    """Compensated Levy exponent  Lambda(k) = int dxi R(xi)(e^{ikxi}-1-ikxi).

    Pass R already tilted (``tilt_source``) for source-plane statistics.
    The -ikxi subtraction is the ensemble-mean subtraction: <xi> = 0.

    ``nproc>1`` splits the (usually 60000-point) k-array across a process
    pool -- this is the single most expensive step in a full PDF build, and
    every k-slice is independent. Default nproc=1 is unchanged/serial.
    """
    k = np.atleast_1d(np.asarray(k, float))
    xif, Rf = _fine_grid(R, nfine)
    if nproc is None or nproc <= 1 or k.size <= chunk:
        return _lambda_k_chunk((k, xif, Rf, chunk))
    import concurrent.futures as _cf
    kchunks = [c for c in np.array_split(k, nproc) if c.size]
    payload = [(c, xif, Rf, chunk) for c in kchunks]
    with _cf.ProcessPoolExecutor(max_workers=min(nproc, len(payload))) as ex:
        parts = list(ex.map(_lambda_k_chunk, payload))
    return np.concatenate(parts)


def _kmax_auto(R, nfine=6000):
    """Spec Sec. 7 recipe: estimate the linear decay of Re Lambda and cut where
    e^{Re Lambda} ~ e^{-35}."""
    kp = np.array([50.0, 200.0])
    L = Lambda_of_k(kp, R, nfine)          # 2 points -- never worth nproc>1
    slope = (L[0].real - L[1].real) / (kp[1] - kp[0])   # >0
    if not np.isfinite(slope) or slope <= 0:
        return 2.0e4
    return float(np.clip(35.0 / slope, 2.0e3, 4.0e4))


def P_of_xi(R, xi_out=None, kmax=None, nk=60000, nfine=6000, nproc=1):
    """Invert to the pdf of xi = ln mu:  P(xi) = (1/pi) Re int_0^inf dk
    e^{-i k xi + Lambda(k)}.  Pass a tilted R for the source-plane PDF.

    ``nproc>1`` parallelizes the dominant ``Lambda_of_k`` step (see there);
    the final per-xi_out quadrature stays serial (cheap: O(nk) per point).
    """
    if xi_out is None:
        xi_out = np.linspace(-0.9, 1.6, 420)
    xi_out = np.asarray(xi_out, float)
    if kmax is None:
        kmax = _kmax_auto(R, nfine)
    kg = np.linspace(0.0, kmax, nk)
    Lam = Lambda_of_k(kg, R, nfine, nproc=nproc)
    Phi = np.exp(Lam)
    P = np.empty(xi_out.size)
    for i, xv in enumerate(xi_out):
        P[i] = _trapz((Phi * np.exp(-1j * kg * xv)).real, kg) / pi
    return xi_out, P


def dP_dmu(R, mu=None, **kw):
    """Source-plane magnification pdf.  dP/dmu = P(ln mu)/mu."""
    if mu is None:
        mu = np.linspace(0.5, 4.0, 400)
    xi = log(np.asarray(mu, float))
    _, P = P_of_xi(R, xi_out=xi, **kw)
    return mu, P / mu


# ============================== moments =====================================
def Ltilde(R, t):
    """Real-argument exponent  int R (e^{-t xi} - 1 + t xi) dxi (exact
    moments; pass a tilted R for source-plane)."""
    return _trapz(R * (exp(-t * XI) - 1 + t * XI), XI)


def sigma_DL_over_DL(R):
    """Exact fractional distance scatter, D_L ~ mu^{-1/2}:
    (sigma/D)^2 = exp[Ltilde(1) - 2 Ltilde(1/2)] - 1."""
    return sqrt(exp(Ltilde(R, 1.0) - 2 * Ltilde(R, 0.5)) - 1.0)


def moments(R):
    """Cumulants of xi for the compensated process: var, skew, kurtosis."""
    m2 = _trapz(R * XI**2, XI)
    m3 = _trapz(R * XI**3, XI)
    m4 = _trapz(R * XI**4, XI)
    return dict(var=m2, skew=m3 / m2**1.5, kurt=m4 / m2**2)


def slope_R(R):
    return -np.gradient(log(R), _lnXI)


# ==================== Vector compound-Poisson framework =====================

def build_lens_population(cos, zs, kappa_low=1e-4, Mmin=1e7, Mmax=1e16, Nz=40, NM=48):
    """Discretize the lens population into (kappa, gamma, expected-count) cells.

    Returns (K, G, W, faint_k2, faint_g2) where K, G, W are 1D arrays for cells
    with kappa > kappa_low, and faint_k2, faint_g2 are the Campbell variances of
    the faint unresolved tail (which behaves as an exact Gaussian background).
    """
    zgrid = np.linspace(1e-3, zs - 1e-3, Nz)
    dz = zgrid[1] - zgrid[0]
    Mgrid = np.logspace(np.log10(Mmin), np.log10(Mmax), NM)
    dlnM = log(Mgrid[1]) - log(Mgrid[0])
    K, G, W = [], [], []
    faint_k2 = faint_g2 = 0.0
    xa, xb = _XG[:-1], _XG[1:]
    xm = sqrt(xa * xb)
    for z in zgrid:
        wz = (1 + z)**2 * CKMS / cos.Hz(z) * dz
        for M in Mgrid:
            C, rs, ks, fC = cos.nfw_params(M, z, zs)
            k, g = kappa_gamma(xm, ks)
            w = (wz * cos.dndlnM(M, z) * dlnM
                 * pi * rs * rs * (xb**2 - xa**2))     # expected count
            hi = k > kappa_low
            K.append(k[hi])
            G.append(g[hi])
            W.append(w[hi])
            faint_k2 += float(np.sum(w[~hi] * k[~hi]**2))
            faint_g2 += float(np.sum(w[~hi] * g[~hi]**2))
    return (np.concatenate(K), np.concatenate(G), np.concatenate(W),
            faint_k2, faint_g2)


def _lambda_bins(K, G, W, n_k_bins=400, n_g_bins=250):
    """2D log-binning of the active (kappa, gamma) cells -> (w_b, kap_b, gam_b)."""
    k_edges = np.geomspace(1e-4, 30.0, n_k_bins + 1)
    g_edges = np.geomspace(1e-6, 2.0, n_g_bins + 1)
    H, _, _ = np.histogram2d(K, G, bins=[k_edges, g_edges], weights=W)
    k_m = sqrt(k_edges[:-1] * k_edges[1:])
    g_m = sqrt(g_edges[:-1] * g_edges[1:])
    i_k, i_g = np.nonzero(H)
    return H[i_k, i_g], k_m[i_k], g_m[i_g]


def Lambda_vector(k_k_arr, k_g_arr, K, G, W, faint_k2, faint_g2,
                  n_k_bins=400, n_g_bins=250, chunk=96):
    """Compensated 2D characteristic exponent Lambda(k_kappa, k_gamma).

    Lambda(k_k, k_g) = int d^3q R(q) [ e^{i k_k kappa} J0(k_g gamma) - 1 - i k_k kappa ]

    Evaluated using 2D log-binning of active (kappa, gamma) cells.  The bin sum
    is a single matrix product, sum_b w_b e^{i k_k kappa_b} J0(k_g gamma_b) =
    (E * w) @ J, chunked over k_kappa to bound memory.  It used to be a Python
    loop over k_gamma that rebuilt an (nk_k x nbins) complex temporary on every
    iteration -- same flops, ~2 orders of magnitude more overhead, which is what
    made the reach this exponent needs (see `phi_reach`) unaffordable.
    """
    import scipy.special as sp

    w_b, k_b, g_b = _lambda_bins(K, G, W, n_k_bins, n_g_bins)
    k_k_arr = np.asarray(k_k_arr, float)
    k_g_arr = np.asarray(k_g_arr, float)
    nk_k, nk_g = len(k_k_arr), len(k_g_arr)

    J = sp.j0(np.outer(g_b, k_g_arr))            # (nbins, nk_g), real
    Lambda = np.empty((nk_k, nk_g), dtype=complex)
    for lo in range(0, nk_k, chunk):
        hi = min(lo + chunk, nk_k)
        ph = np.outer(k_k_arr[lo:hi], k_b)       # (c, nbins)
        Lambda[lo:hi] = (np.exp(1j * ph) * w_b) @ J
    # the -1 and -i k_k kappa compensation terms are k_gamma-independent
    Lambda -= (w_b.sum() + 1j * np.outer(k_k_arr, w_b @ k_b))
    Lambda -= 0.5 * np.outer(k_k_arr**2 * faint_k2, np.ones(nk_g))
    Lambda -= 0.25 * np.outer(np.ones(nk_k), k_g_arr**2 * faint_g2)
    return Lambda


def P_vector_joint(k_k_arr, k_g_arr, Phi, kap_arr, gam_arr):
    """Invert Phi(k_kappa, k_gamma) to 3D isotropic PDF P(kappa, gamma_1, gamma_2) = P(kappa, gamma).

    1. Hankel transform over k_gamma:
       F(k_kappa, gamma) = int_0^inf k_gamma dk_gamma J0(k_gamma gamma) Phi(k_kappa, k_gamma)
    2. Inverse Fourier transform over k_kappa:
       P(kappa, gamma) = (1 / 2pi^2) Re int_0^inf dk_kappa e^{-i k_kappa kappa} F(k_kappa, gamma)
    """
    import scipy.special as sp

    nk_k = len(k_k_arr)
    nk_g = len(k_g_arr)
    dk_k = k_k_arr[1] - k_k_arr[0]
    dk_g = k_g_arr[1] - k_g_arr[0]

    # Hankel transform
    J0_mat = sp.j0(np.outer(gam_arr, k_g_arr))
    w_kg = np.full(nk_g, dk_g)
    w_kg[0] *= 0.5
    w_kg[-1] *= 0.5
    kg_weights = k_g_arr * w_kg
    F = Phi @ (kg_weights[:, None] * J0_mat.T)

    # Fourier transform
    w_kk = np.full(nk_k, dk_k)
    w_kk[0] *= 0.5
    w_kk[-1] *= 0.5
    exp_kap = np.exp(-1j * np.outer(kap_arr, k_k_arr))
    P_3d = (1.0 / (2.0 * pi**2)) * np.real(exp_kap @ (F * w_kk[:, None]))
    return P_3d


# --- width-relative grid rules for the vector inversion ---------------------
# ⚠ Every grid below USED to be a hardcoded absolute number (nk_k=256,
# nk_g=200, k_k_max=160, k_g_max=200, kap_min=-0.45, kap_max=0.50,
# gam_max=0.70).  sigma_kappa of the halo population runs 0.0124 (z_s=0.5) to
# 0.1106 (z_s=10) -- 9x -- so a fixed k-space reach is 5x too short at z_s=0.5
# and a fixed kappa box holds only 3.7 points per sigma there.  The k_kappa
# integral was then truncated before Phi(k) had decayed and the inversion rang:
# at z_s=1 it returned Var = 6.1e-4 and skew = -3.1 against the MC's 2.70e-3
# and +3.79.  It happened to be adequate for z_s >~ 2, which is why the high-z
# panels looked right.  Same failure class as the emulator's global lnmu
# standardization and its fixed-dlnmu acceptance panel (see CLAUDE.md): an
# absolute grid against a width that spans orders of magnitude.
#
# The grids are therefore set as multiples of sigma = sigma_kappa (numerically
# equal to sigma_gamma for these profiles), which makes them z_s- and
# cosmology-independent in scaled units.  Constants calibrated on the z_s=5
# convergence ladder and re-verified by doubling at z_s = 0.5, 1, 10.
SIG_REACH = 12.0        # STARTING k-space reach, in 1/sigma -- refined by phi_reach
PHI_TOL = 1e-10         # |Phi| at the k-space cut; sets the Gibbs ringing floor
PPP_K = 18.0            # quadrature points per period of exp(-i k_k kappa)
PPP_G = 96.0            # quadrature points per period of J0(k_g gamma)
BOX_REACH = 6.0         # (kappa, gamma) box half-extent, in sigma
PTS_PER_SIG_KAP = 24.0  # kappa-grid resolution
PTS_PER_SIG_GAM = 12.0  # gamma-grid resolution
XI_REACH = 24.0         # moment/normalization grid reach, in sigma_xi = 2 sigma
# ^ 24 not 8: the magnification tail carries negligible MASS but a large share
#   of the MOMENTS.  At z_s=5 a reach of 8 cuts only 7.5e-5 of the probability
#   yet loses 2.9% of Var and 15% of the skew; the ladder 8/16/24/32 reads
#   var 0.026011/0.026733/0.026790/0.026796 and skew 1.591/1.826/1.863/1.869,
#   i.e. converged by 24 (32 moves var 5e-3% and skew 0.3%).  The PLOTTED shape
#   is insensitive to this (<=0.04% at z_s>=5, <=1% at z_s<=1) -- it is the
#   moments that need the reach.


def phi_reach(K, G, W, faint_k2, faint_g2, k0, axis, tol=PHI_TOL, max_mult=64.0):
    """Smallest k on `axis` ("k" or "g") at which |Phi| has fallen to `tol`.

    ⚠ The reach CANNOT be derived from sigma, which is what an earlier version
    of this module assumed (k_max = SIG_REACH/sigma, justified by
    Phi ~ exp(-k^2 sigma^2/2)).  That reasoning is wrong twice over: the only
    Gaussian term in Lambda carries the FAINT band's variance, not the total
    (faint_k2/sigma^2 ~ 2e-3 here, so at k = 12/sigma it damps by exp(-0.19) --
    nothing), and the actual decay is Riemann-Lebesgue decorrelation of the
    explicit Poisson sum, whose rate goes with the LENS COUNT.  Measured |Phi|
    at k = 12/sigma ran 7.1e-3 (z_s=0.5, N=38) to 3.9e-8 (z_s=10, N=1946) --
    five orders of magnitude at a nominally fixed reach -- and that residual is
    exactly the Fourier ringing seen in the z_s <= 1 panels.  So measure it.

    Returns (k_max, phi_at_cut, atom_weight).  `atom_weight` = e^{-N} is the
    compound-Poisson "no explicit lens" atom: |Phi| can never fall below it, so
    if it exceeds `tol` the density has a genuine unresolvable spike and NO
    reach removes the ringing -- the caller is told rather than left to wonder.
    """
    atom = float(exp(-W.sum()))
    mults = 1.25 ** np.arange(0, int(np.ceil(log(max_mult) / log(1.25))) + 1)
    ks = k0 * mults
    zero = np.zeros(1)
    if axis == "k":
        L = Lambda_vector(ks, zero, K, G, W, faint_k2, faint_g2)[:, 0]
    else:
        L = Lambda_vector(zero, ks, K, G, W, faint_k2, faint_g2)[0, :]
    phi = np.abs(np.exp(L))
    ok = np.nonzero(phi <= tol)[0]
    i = int(ok[0]) if len(ok) else int(np.argmin(phi))
    return float(ks[i]), float(phi[i]), atom


def vector_grid(sigma, xi_reach=XI_REACH):
    """The width-relative inversion grid for a population of width ``sigma``.

    Returned as a plain dict so a caller can print it, log it, or override any
    single entry.  ``kap_max`` is tied to the moment grid's ceiling ``xi_max``
    through mu = e^{xi}, kappa_upper = 1 - e^{-xi/2}, so the box cannot silently
    truncate the tail the moments are integrated over -- the failure the fixed
    kap_max=0.50 / xi_wide=(-0.9, 1.6) pair had (it zeroed everything above
    mu=4 while still integrating to mu=4.95).
    """
    sig_xi = 2.0 * sigma
    xi_max = xi_reach * sig_xi
    xi_min = -BOX_REACH * sig_xi
    kap_max = 1.0 - exp(-0.5 * xi_max)
    kap_min = -BOX_REACH * sigma
    gam_max = BOX_REACH * sigma
    k_k_max = SIG_REACH / sigma
    k_g_max = SIG_REACH / sigma
    kap_ext = max(abs(kap_min), kap_max)
    dk_k = 2.0 * pi / (PPP_K * kap_ext)
    dk_g = 2.0 * pi / (PPP_G * gam_max)
    return dict(
        nk_k=int(np.ceil(k_k_max / dk_k)) + 1,
        nk_g=int(np.ceil(k_g_max / dk_g)) + 1,
        k_k_max=k_k_max, k_g_max=k_g_max,
        n_kap=int(np.ceil((kap_max - kap_min) / (sigma / PTS_PER_SIG_KAP))) + 1,
        n_gam=int(np.ceil(gam_max / (sigma / PTS_PER_SIG_GAM))) + 1,
        kap_min=kap_min, kap_max=kap_max, gam_max=gam_max,
        xi_min=xi_min, xi_max=xi_max, sigma=sigma,
    )


def P_vector_lnmu(cos, zs, xi_out=None, kappa_low=1e-4,
                  nk_k=None, nk_g=None, k_k_max=None, k_g_max=None,
                  n_kap=None, n_gam=None, kap_min=None, kap_max=None,
                  gam_max=None, n_k_int=300, xi_reach=XI_REACH, n_xi_wide=500,
                  grid_scale=1.0):
    """Compute the source-plane magnification PDF P_s(ln mu) using the vector
    compound-Poisson framework.

    Every grid argument defaults to ``None`` = the width-relative rule in
    ``vector_grid`` (see the note above); pass a number to override just that
    one.  ``grid_scale`` multiplies every resolution at once, which is the
    handle a convergence check should turn.

    Returns (xi_out, P_s, P_i, meta); ``meta["grid"]`` records what was used
    and ``meta["tail_mass_above_xi_max"]`` the source-plane mass the moment
    grid truncates.
    """
    from scipy.interpolate import RegularGridInterpolator

    if xi_out is None:
        xi_out = np.linspace(-0.35, 0.55, 160)
    xi_out = np.asarray(xi_out, float)

    K, G, W, fk2, fg2 = build_lens_population(cos, zs, kappa_low=kappa_low)

    sigma = float(sqrt(np.sum(W * K**2) + fk2))
    grid = vector_grid(sigma, xi_reach=xi_reach)
    # Reach is MEASURED, not assumed -- see phi_reach.  Skipped when the caller
    # pins the reach explicitly.
    if k_k_max is None:
        grid["k_k_max"], phi_k, atom = phi_reach(K, G, W, fk2, fg2,
                                                 grid["k_k_max"], "k")
    else:
        phi_k, atom = float("nan"), float(exp(-W.sum()))
    if k_g_max is None:
        grid["k_g_max"], phi_g, _ = phi_reach(K, G, W, fk2, fg2,
                                              grid["k_g_max"], "g")
    else:
        phi_g = float("nan")
    # dk is set by the box extent, so re-derive the point counts at the new reach
    kap_ext = max(abs(grid["kap_min"]), grid["kap_max"])
    grid["nk_k"] = int(np.ceil(grid["k_k_max"]
                               / (2.0 * pi / (PPP_K * kap_ext)))) + 1
    grid["nk_g"] = int(np.ceil(grid["k_g_max"]
                               / (2.0 * pi / (PPP_G * grid["gam_max"])))) + 1
    grid.update(phi_at_k_cut=phi_k, phi_at_g_cut=phi_g, atom_weight=atom)
    if atom > PHI_TOL:
        print(f"[warn] P_vector_lnmu(zs={zs:g}): no-lens atom e^-N = {atom:.2e} "
              f"exceeds PHI_TOL = {PHI_TOL:.0e}; the density has a genuine "
              "unresolvable spike and Fourier ringing cannot be removed by reach",
              file=__import__("sys").stderr)
    for key, val in dict(nk_k=nk_k, nk_g=nk_g, k_k_max=k_k_max,
                         k_g_max=k_g_max, n_kap=n_kap, n_gam=n_gam,
                         kap_min=kap_min, kap_max=kap_max,
                         gam_max=gam_max).items():
        if val is not None:
            grid[key] = val
    if grid_scale != 1.0:
        for key in ("nk_k", "nk_g", "n_kap", "n_gam"):
            grid[key] = int(np.ceil(grid[key] * grid_scale))
    nk_k, nk_g = grid["nk_k"], grid["nk_g"]
    k_k_max, k_g_max = grid["k_k_max"], grid["k_g_max"]
    n_kap, n_gam = grid["n_kap"], grid["n_gam"]
    kap_min, kap_max, gam_max = grid["kap_min"], grid["kap_max"], grid["gam_max"]

    k_k_arr = np.linspace(0.0, k_k_max, nk_k)
    k_g_arr = np.linspace(0.0, k_g_max, nk_g)

    Lambda = Lambda_vector(k_k_arr, k_g_arr, K, G, W, fk2, fg2)
    Phi = np.exp(Lambda)

    kap_arr = np.linspace(kap_min, kap_max, n_kap)
    gam_arr = np.linspace(0.0, gam_max, n_gam)

    P_3d = P_vector_joint(k_k_arr, k_g_arr, Phi, kap_arr, gam_arr)
    interp_P = RegularGridInterpolator((kap_arr, gam_arr), P_3d,
                                       bounds_error=False, fill_value=0.0)

    def _eval_P_i(xi_grid):
        p_mu = np.zeros_like(xi_grid)
        for i, xi in enumerate(xi_grid):
            mu = exp(xi)
            kap_upper = 1.0 - 1.0 / sqrt(mu)
            # Only kappa within ~gam_max of the gamma = 0 point contributes:
            # gamma(kappa, mu) <= gam_max  <=>  kappa >= 1 - sqrt(1/mu +
            # gam_max^2).  The old fixed 0.45 was ~36 sigma at z_s=0.5 and
            # ~5 sigma at z_s=10, so it wasted resolution at one end and
            # clipped the integrand at the other.
            kap_lower = max(kap_arr[0], 1.0 - sqrt(1.0 / mu + gam_max**2))
            if kap_upper <= kap_lower:
                continue
            k_eval = np.linspace(kap_lower, kap_upper, n_k_int)
            rad = np.clip((1.0 - k_eval)**2 - 1.0 / mu, 0.0, None)
            g_eval = sqrt(rad)
            pts = np.column_stack([k_eval, g_eval])
            integral = _trapz(interp_P(pts), k_eval)
            p_mu[i] = (pi / (mu * mu)) * integral
        return p_mu

    # Global moments over wide support -- width-relative, and tied to kap_max
    # so the box cannot zero part of the range the moments integrate over.
    xi_wide = np.linspace(grid["xi_min"], grid["xi_max"], n_xi_wide)
    p_wide = _eval_P_i(xi_wide)
    norm_w = _trapz(p_wide, xi_wide)
    ps_wide = p_wide / norm_w if norm_w > 0 else p_wide
    mean_s = float(_trapz(xi_wide * ps_wide, xi_wide))
    var_s = float(_trapz((xi_wide - mean_s)**2 * ps_wide, xi_wide))
    skew_s = float(_trapz((xi_wide - mean_s)**3 * ps_wide, xi_wide) / var_s**1.5) if var_s > 0 else 0.0
    x_dist = exp(-0.5 * xi_wide)
    mean_d = float(_trapz(x_dist * ps_wide, xi_wide))
    var_d = float(_trapz((x_dist - mean_d)**2 * ps_wide, xi_wide))
    sigma_DL = float(sqrt(var_d) / mean_d) if mean_d > 0 else 0.0

    # Evaluate on requested xi_out
    P_i_mu = _eval_P_i(xi_out)
    P_s = P_i_mu / norm_w if norm_w > 0 else P_i_mu

    mu_arr = exp(xi_out)
    P_i_xi = mu_arr * P_i_mu
    norm_i = _trapz(P_i_xi, xi_out)
    if norm_i > 0:
        P_i_xi /= norm_i

    # Source-plane mass beyond the moment grid: the honest size of the
    # truncation every var/skew/sigma_DL below carries.
    tail_hi = _eval_P_i(np.linspace(grid["xi_max"],
                                    grid["xi_max"] + 4.0 * sigma, 40))
    meta = dict(
        grid=grid,
        sigma_kappa=sigma,
        tail_mass_above_xi_max=float(
            _trapz(tail_hi, np.linspace(grid["xi_max"],
                                        grid["xi_max"] + 4.0 * sigma, 40))
            / norm_w) if norm_w > 0 else float("nan"),
        var_k=float(np.sum(W * K**2) + fk2),
        var_g=float(np.sum(W * G**2) + fg2),
        N_tot=float(W.sum()),
        mean=mean_s,
        var=var_s,
        skew=skew_s,
        sigmaDL=sigma_DL,
        interp_P=interp_P,
        kap_min=kap_min,
        kap_max=kap_max,
    )
    return xi_out, P_s, P_i_xi, meta
