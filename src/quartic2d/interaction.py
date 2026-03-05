import numpy as np

from numpy.fft import fft
from scipy.interpolate import CubicSpline
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.integrate import cumulative_trapezoid
from matplotlib import pyplot as plt

import scipy.integrate as integrate
import hankel
import orbitalspectrum

def _phi_positive(s):
    # phi(s) = exp(-1/s) for s>0, 0 otherwise
    s = np.asarray(s)
    out = np.zeros_like(s, dtype=float)
    mask = s > 0
    out[mask] = np.exp(-1.0 / s[mask])
    return out

def smooth_bump(r, r_i, r_f):
    """
    C^infty bump: equals 1 at r=r_i (and flat there), smoothly transitions on (r_i,r_f),
    and equals 0 for r >= r_f (flat at r_f).
    """
    r = np.asarray(r)
    t = (r - r_i) / (r_f - r_i)   # t in (-inf, +inf)
    S = np.zeros_like(t, dtype=float)

    inside = (t > 0) & (t < 1)

    if np.any(inside):
        tt = t[inside]
        p_t = _phi_positive(tt)
        p_1mt = _phi_positive(1.0 - tt)
        S[inside] = p_t / (p_t + p_1mt)

    S[t >= 1.0] = 1.0
    S[t <= 0.0] = 0.0

    bump = 1.0 - S   # bump(r_i)=1, bump(r_f)=0, flat at both ends
    return bump
def cinf_bump(r: np.ndarray, r_start: float, r_end: float) -> np.ndarray:
    """
    C^infty bump: equals 1 at r=r_i (and flat there), smoothly transitions on (r_i,r_f),
    and equals 0 for r >= r_f (flat at r_f).
    """
    r = np.asarray(r)
    t = (r - r_start) / (r_end - r_start)   # t in (-inf, +inf)
    S = np.zeros_like(t, dtype=float)

    inside = (t > 0) & (t < 1)

    if np.any(inside):
        tt = t[inside]
        p_t = _phi_positive(tt)
        p_1mt = _phi_positive(1.0 - tt)
        S[inside] = p_t / (p_t + p_1mt)

    S[t >= 1.0] = 1.0
    S[t <= 0.0] = 0.0

    bump = 1.0 - S   # bump(r_i)=1, bump(r_f)=0, flat at both ends
    return bump
    

def zero_pad_radial_tail( r: np.ndarray,
                         f: np.ndarray,
                         pad: float = 2.0,
                         N_pad: int = 1024,
                         force_r0: bool = True
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero-pad the radial profile beyond its last radius to reduce window-convolution ringing
    in the Hankel transform.

    Parameters
    ----------
    r : 1D np.array
        Monotonic increasing radii in [0, r_cut].
    f : 1D np.array
        Values on r (e.g., rho_m(r) possibly already tapered).
    pad : float
        Extend the radius to r_pad_max = pad * r[-1], filled with zeros.
        Typical 2.0–3.0 is enough to noticeably narrow the q-space convolution kernel.
    N_pad : int
        Number of zero samples to append between (r[-1], r_pad_max].
    force_r0 : bool
        If True and r[0] > 0, prepend a point at r=0 with f=f[0] so that
        the spline has a knot at the origin (often stabilizes m=0).

    Returns
    -------
    r_ext, f_ext : 1D arrays
        Extended grids with zeros beyond r[-1].
    """
    r0 = r[0]
    r_cut = r[-1]
    r_pad_max = float(pad) * r_cut

    # optional: ensure a knot at r=0 (helps m=0 stability if original grid starts at small r>0)
    if force_r0 and r0 > 0:
        r = np.concatenate(([0.0], r))
        f = np.concatenate(([f[0]], f))

    if N_pad > 0 and r_pad_max > r_cut:
        r_tail = np.linspace(r_cut, r_pad_max, N_pad + 1)[1:]  # exclude r_cut duplicate
        f_tail = np.zeros_like(r_tail, dtype=f.dtype)
        r_ext = np.concatenate((r, r_tail))
        f_ext = np.concatenate((f, f_tail))
    else:
        r_ext, f_ext = r, f

    # guard: unique & np.sorted (spline requires strictly increasing x)
    r_ext, uniq_idx = np.unique(r_ext, return_index=True)
    f_ext = f_ext[uniq_idx]
    return r_ext, f_ext

class HankelTransform():

    def __init__(self,
                 nu: int,
                 q: np.ndarray, 
                 F_r: np.ndarray,
                 r: np.ndarray,
                 N: int, 
                 r_cutoff,
                 h: float | None = None,
                 tol_power: float = 1e-10,
                 quad_tol: float = 1e-4,
                 max_subdivs: int = 100,
                 interp_method: str = 'linear',
                 eta_bump: float | None = None, 
                 zeropadparams: tuple | None = None,):
        
        self._F_r = F_r
        self._r = r
        self._r_cutoff = r_cutoff
        self._q = q
        
        self._nu = nu
        self._N = N

        self._quad_tol = quad_tol
        self._max_subdivs = max_subdivs

        # Restrict r-domain up to r_cut
        r_mask = r <= r_cutoff
        self._r, self._F_r = r[r_mask], self._F_r[r_mask]
        
        # Set radial step for Hankel transform: h = pi / N
        if h is None:
            self._h = np.pi / N
        else:
            self._h = h
            
        self._method = interp_method

        self._precompute(eta_bump, zeropadparams, tol_power)

    def _add_cinf_bump(self, eta):
        r_start = (1.0 - eta) * self._r_cutoff
        w = cinf_bump(self._r, r_start, self._r_cutoff)
        self._F_r *= w 

    def _add_zero_pad_tail(self, pad = 3.0, N_pad = 1028):
        self._r, self._F_r = zero_pad_radial_tail(self._r, self._F_r, pad = pad, N_pad = N_pad, force_r0 = True)

    def _interpolate(self, method):

        r = self._r
        F = self._F_r

        if method == "linear":

            def f_r(x):
                x = np.asarray(x)
                return np.interp(x, r, F, left=0.0, right=0.0)

            self._interpolation = f_r

        elif method == "cubic":

            spline = CubicSpline(r, F, bc_type="natural", extrapolate=False)

            def f_r(x):
                x = np.asarray(x)
                out = spline(x)
                out[np.isnan(out)] = 0.0
                return out

            self._interpolation = f_r

        else:
            raise ValueError("method must be 'linear', or 'cubic'")

    def _calculate_q_cutoff(self, tol_power = 1e-10):
        
        power_density = np.abs(self.F_q) ** 2 * self._q             # q * |F(q)|²

        cumulative_power = cumulative_trapezoid(power_density, self._q, initial = 0)
        total_power = cumulative_power[-1]

        # Target power fraction cutoff: np.where cumulative power reaches 99.9% (default)
        target = (1 - tol_power) * total_power
        idx = np.searchsorted(cumulative_power, target)
        
        self.q_cutoff = float(self._q[idx]) if idx < len(self._q) else float(self._q[-1])

    def _precompute(self, eta_bump, zeropadparams, tol_power):
        
        # Initialize Hankel transform for order ν
        self._ht = hankel.HankelTransform(nu = np.abs(self._nu), N = self._N, h = self._h)

        if eta_bump is not None:
            self._add_cinf_bump(eta_bump)

        # zero-pad to extend window (narrows convolution kernel in q)
        if zeropadparams:
            self._add_zero_pad_tail(*zeropadparams)

        self._interpolate(self._method)       

        # Perform Hankel transform:
        sign = (-1.0)**(np.abs(self._nu)) if self._nu < 0 else 1

        self.F_q = np.zeros_like(self._q, dtype = np.cdouble)

        zero_mask = np.isclose(self._q, 0)
        nonzero_mask = np.invert(zero_mask)
        
        self.F_q[nonzero_mask] = sign * self._ht.transform(self._interpolation, self._q[nonzero_mask], ret_err = False)
        # self.F_q, self.error = self._ht.transform(self._interpolation, self._q)
        if zero_mask.any():
            if self._nu == 0:
                kwargs = {'epsabs': self._quad_tol,
                        'limit': self._max_subdivs,
                        'complex_func': True}
                g = lambda q: q * self._interpolation(q)
                self.F_q[zero_mask], _ = integrate.quad(g, a = 0.0, b = np.inf, **kwargs)
            else:
                self.F_q[zero_mask] = 0

        self._calculate_q_cutoff(tol_power = tol_power)
        
        # valid = qg <= q_cut[m]

        self.spline = RGI((self._q, ), self.F_q,  bounds_error = False, fill_value = 0.0)
        # spline_q = RGI((qg[valid],), Fq[valid],)

class HankelofHarmonics():
    
    def __init__(self, cdh: orbitalspectrum.PolarDecomposition, 
                 q_max: float, 
                 N: int = 2048, 
                 h: float | None = None,
                 N_q: int = 1024,
                 interp_method: str = 'linear',
                 eta_bump: float = None,
                 zeropadparams: tuple | None = None, 
                 tol_power = 1e-10,
                 tol_tot_error: float | None = None):
        """
        Compute the Hankel transforms of all angular-harmonic radial components ρ_m(r),
        and define a physically motivated cutoff momentum q_cut for each harmonic based
        on the *cumulative q-space power*.

        The output includes smooth spline interpolants F_m(q) that are zero beyond q_cut.

        --------------------------------------------------------------------------
        Parameters
        --------------------------------------------------------------------------
        results : dict
            The output from an angular harmonic decomposition (either discrete or continuous).
            Must contain:
                - 'r'               : radial grid np.array (1D)
                - 'rho_m_r'         : np.array (Nr × Nm) of radial harmonics ρ_m(r)
                - 'm_vals_mmax'     : list/np.array of all m-values in the decomposition
                - 'rcut_quantile'   : cutoff radii per harmonic
                - (optional) 'm_vals_auto' : automatically selected m's

        mode : {'auto', 'm_max'}, default 'auto'
            Whether to use only automatically selected m-values or all m-values up to m_max.

        N : int, default 1024
            Number of subdivisions in the Hankel transform. Larger N = finer sampling
            but heavier computation. The step size in r-space is set as h = r_cut / N.

        quantile : float or None
            Fraction of cumulative radial power used to define the *r-space* cutoff (r_cut).
            If None, uses the same quantile already stored in `results['quantile']`.

        power_fraction_q : float, default 0.999
            Fraction of total *q-space power* defining q_cut. For example, 0.999 means
            we retain 99.9% of the power ∫ q|F_m(q)|² dq before truncating.

        spline_order : int, default 5
            Polynomial order of the spline interpolation for F_m(q). High order (e.g. 5)
            gives smoother, more accurate interpolation than linear.

        --------------------------------------------------------------------------
        Returns
        --------------------------------------------------------------------------
        out : dict
            {
            'm_list'        : np.array of int, list of selected harmonics
            'q_grid'        : dict[int → np.ndarray], momentum grids used per harmonic
            'F_m'           : dict[int → np.ndarray], raw Hankel transform arrays
            'F_m_spline'    : dict[int → callable], smooth interpolants F_m(q)
            'q_cut'         : dict[int → float], cutoff momenta per harmonic
            'global_q_cut'  : float, largest q_cut across all harmonics
            'power_fraction_q': float, same as input
            }

        --------------------------------------------------------------------------
        Notes
        --------------------------------------------------------------------------
        • Uses order |m| for each Hankel transform.
        • No tapering at r_cut: we simply zero-extend the spline beyond r_cut.
        • The spline interpolants return 0 beyond q_cut for safety.
        • The "HankelTransform" convention determines normalization.
        """ 

        self._cdh = cdh
        self._rho = cdh.rho
        self._r = cdh.r
        self._m_vals = cdh.m_sorted
        self._r_cutoffs = cdh._r_cutoff
        self._q_max = q_max

        self._N_q = N_q
        self._N = N
        self._h = h
        self._method = interp_method
        self._tol_power = tol_power
        self._eta_bump = eta_bump
        self._zeropadparams = zeropadparams
        

        self._compute()
        if tol_tot_error is not None:
            self._check_error(tol_tot_error)

    def __call__(self, m, q):
        q = np.array(q)
        if q.ndim == 2:
            f = self.F_q_interp[m](q.flatten())
            return f.reshape(q.shape)
        q = q.reshape(-1, 1)
        return self.F_q_interp[m](q)

    def _compute(self):

        self.F_q, self.F_q_interp, self.q, self.q_cutoffs = {}, {}, {}, {}
        
        for i, m in enumerate(self._m_vals):

            if self._h is None:
                h = np.pi/self._N
            else:
                h = self._h
            
            # rc = float(r_cut_all[j])    # cutoff radius 
            
            # Extract the q-grid from the hankel object (may be 'k' or 'q' depending on version)
            q = np.linspace(0, self._q_max, self._N_q)

            # plt.plot(self._cdh[m], label = f'{m}')
            # plt.legend()
            # plt.show()

            HT = HankelTransform(m, 
                                 q, 
                                 self._cdh[m], 
                                 self._r, 
                                 self._N, 
                                 self._r_cutoffs[i], 
                                 h = h, 
                                 interp_method = self._method,
                                 eta_bump = self._eta_bump, 
                                 zeropadparams = self._zeropadparams,
                                 tol_power = self._tol_power)

            self.F_q[m] = HT.F_q
            self.q[m] = q
            self.F_q_interp[m] = HT.spline
            self.q_cutoffs[m] = HT.q_cutoff

    def _check_error(self, tol):
        
        tot_error_sqr = 0

        zero_mask = np.isclose(self._r, 0)
        nonzero_mask = np.invert(zero_mask)

        self.rho_inverse = {}
                
        for m in self._m_vals:

            self.rho_inverse[m] = np.zeros_like(self._r, dtype = np.cdouble)
            
            nu = np.abs(m)
            sign = (-1.0)**(nu) if m < 0 else 1.0
            
            ht = hankel.HankelTransform(nu, self._N, np.pi/self._N)
            
            def F_q(m, q):
                q = np.array(q)
                if q.ndim == 2:
                    f = self.F_q_interp[m](q.flatten())
                    return f.reshape(q.shape)
                q = q.reshape(-1, 1)
                return self.F_q_interp[m](q)
            
            F = lambda q: F_q(m, q)
            self.rho_inverse[m][nonzero_mask] = sign * ht.transform(F, self._r[nonzero_mask], ret_err = False)
            if zero_mask.any():

                if m == 0:
                    kwargs = {'epsabs': 1e-4,
                            'limit': 200,
                            'complex_func': True}
                    G = lambda q: q * F(q)
                    self.rho_inverse[m][zero_mask], _ = integrate.quad(G, a = 0.0, b = np.inf, **kwargs)
                else:
                    self.rho_inverse[m][zero_mask] = 0

            roundtrip_error = np.linalg.norm(self._rho[m] - self.rho_inverse[m]) / np.linalg.norm(self._rho[m])
            tot_error_sqr += roundtrip_error**2 * self._cdh.power_fracs[m]
        
        self.total_error = np.sqrt(tot_error_sqr)*100

        if self.total_error > tol:
            raise Warning(f'The roundtrip error, {self.total_error:.4f}%, exceeds tolerance of {tol}%.')

    def plot_hankel_transforms(self, title: str = "Hankel-transformed harmonics"):
        """
        Visualize the Hankel transforms F_m(q) of the radial harmonic components
        and mark each harmonic's q_cutoff with a vertical line.

        Parameters
        ----------
        HT : dict
            Output from `precompute_hankel_of_harmonics_power_cut`.
            Must contain:
                - 'm_list'  : np.array of m-values
                - 'F_m'     : dict[int -> np.ndarray], Hankel-transformed radial profiles
                - 'q_grid'  : dict[int -> np.ndarray], q grids used per harmonic
                - 'q_cut'   : dict[int -> float],  cutoff momenta per harmonic
                - 'power_fraction_q' : float, fraction used for cutoff definition
        max_profiles : int, default 8
            Maximum number of harmonics to plot (most important ones if >8).
        logy : bool, default False
            Plot |F_m(q)| on logarithmic scale if True.
        title : str, default "Hankel-transformed harmonics"
            Figure title.
        color_map : matplotlib colormap
            Colormap used to assign colors to different harmonics.

        Notes
        -----
        • Each harmonic F_m(q) is plotted as |F_m(q)| vs q.
        • Vertical dashed line marks q_cutoff where 99.9% (by default) of cumulative power is included.
        • Uses linear or log-y plotting according to `logy`.
        """
        colors = plt.cm.tab10(np.linspace(0, 1, len(self._m_vals)))
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, m in enumerate(self._m_vals):

            ax.plot(self.q[m], np.abs(self.F_q[m]), label = f"m={m}")
            ax.axvline(self.q_cutoffs[m], ls = "--", c = colors[i], lw = 1.2, alpha = 0.8)

        ax.set_xlabel(r"$q~[\mathrm{\AA}^{-1}]$")
        ax.set_ylabel(r"$|F_m(q)|$")
        ax.set_title(f"{title}")

        ax.legend(fontsize = "small", loc = "best")
        ax.grid(which="both", ls = "--", alpha = 0.4)
        plt.tight_layout()
        plt.show()    

class Interact:

    def __init__(self,
                 deltas: np.ndarray, 
                 h1: HankelofHarmonics,
                 h2: HankelofHarmonics,
                 U_q: callable,
                 N: int | None = 1024,
                 h: float | None = None,
                 delta_zero_tol: float = 1e-4,
                 delta_zero_max_subdivs: int = 100):
        
        self._h1 = h1
        self._h2 = h2
        self._U_q = U_q

        self._deltazero_tol = delta_zero_tol
        self._deltazero_max_subdivs = delta_zero_max_subdivs

        self._delta_mags, self._delta_angles, self._zero_mask = self._delta_mags_and_angles(deltas)

        self._N = N
        if h is not None:
            self._h = h
        else:
            self._h = np.pi / N

        self._mvals1 = np.array(h1._m_vals)
        self._mvals2 = np.array(h2._m_vals)

        self._Phi_mm = self.calculate_Phi_mm()
        self._H_mm = self._calculate_H_mm()

        self._V_mm = 2 * np.pi * self._Phi_mm * self._H_mm
        self._V = self._V_mm.sum(axis = (0, 1))
        self.V = self._V

    def _delta_mags_and_angles(self, deltas):
        
        self._deltas = np.array(deltas)
        
        if deltas.ndim < 2:
            raise ValueError(f'Expected deltas to be 2D array of shape (D, 2) \n with D the number of deltas, found {self._deltas.shape}')
        
        d = np.sqrt(np.sum(self._deltas**2, axis = 1))
        phi = np.arctan2(self._deltas[:,1], self._deltas[:,0])

        zero_mask = np.isclose(d, 0)
        self._containsdeltazero = zero_mask.any()
        
        assert d.shape == (self._deltas.shape[0], )
        assert phi.shape == (self._deltas.shape[0], )
        
        return d, phi, zero_mask

    def calculate_Phi_mm(self):

        m_dif = self._mvals1[:, None] - self._mvals2[None, :]
        assert m_dif.shape == (len(self._mvals1), len(self._mvals2))
        
        Exp = np.exp(1j * m_dif[:, :, None] * self._delta_angles[None, None, :])
        assert Exp.shape == (len(self._mvals1), len(self._mvals2), len(self._delta_angles))

        return Exp


    def _calculate_H_mm(self):

        kwargs = {'epsabs': self._deltazero_tol,
                  'limit': self._deltazero_max_subdivs,
                  'complex_func': True}

        H_mm = np.zeros((len(self._mvals1), len(self._mvals2), len(self._delta_mags)), dtype = np.cdouble)

        nonzero_mask = np.invert(self._zero_mask)

        # hankel_cache = {}

        for i, m in enumerate(self._mvals1):
            for ip, mp in enumerate(self._mvals2):
                nu = np.abs(m - mp)
                
                # if nu not in hankel_cache:
                    # hankel_cache[nu] = hankel.HankelTransform(nu=nu, N=self._N, h=self._h)

                # ht = hankel_cache[nu]
                ht = hankel.HankelTransform(nu=nu, N=self._N, h=self._h)

                f = lambda q: self._h1(m, q) * np.conj(self._h2(mp, q)) * self._U_q(q)
                H_mm[i, ip, nonzero_mask] = ht.transform(f, k = self._delta_mags[nonzero_mask], ret_err = False)

                # handle delta = 0 case
                if self._containsdeltazero:
                    if nu == 0:
                        g = lambda q: q * self._h1(m, q) * np.conj(self._h2(mp, q)) * self._U_q(q)
                        H_mm[i, ip, self._zero_mask], _ = integrate.quad(g, a = 0.0, b = np.inf, **kwargs)
                    else:
                        H_mm[i, ip, self._zero_mask] = 0.0

               
        return H_mm
