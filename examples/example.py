import quartic2d as q2d
import gasp2d as g2d

import numpy as np
from time import time
from matplotlib import pyplot as plt

analyticexample = True
otherexamples   = True

def psi_nx_ny(nx, ny, ax=1.0, ay=1.0):
    from scipy.special import hermite, factorial
    """
    Returns a function ψ(x,y) for given quantum numbers (nx, ny)
    and oscillator lengths (ax, ay).
    """
    Hx = hermite(nx)
    Hy = hermite(ny)
    norm = 1.0 / np.sqrt(np.pi * ax * ay * (2**(nx+ny) * factorial(nx) * factorial(ny)))
    def psi(x, y):
        return norm * Hx(x/ax) * Hy(y/ay) * np.exp(-0.5*((x/ax)**2 + (y/ay)**2))
    return psi

def U_q(q):
    return 2*np.pi/q

#---------------------------------------------------------------------------------------
#          Example with Analytic Solution: Interaction Energy between Gaussians
#---------------------------------------------------------------------------------------

if analyticexample:
    print('#'*100)
    print(f'Analytic Example: Interaction energy between Isotropic Gaussians')
    print('#'*100 + "\n")
    Nx = Ny = 101

    x = np.linspace(-10, 10, Nx) 
    y = np.linspace(-10, 10, Ny)

    X, Y = np.meshgrid(x, y, indexing = 'ij')

    QHO_Ground_State = psi_nx_ny(0, 0, ax = 1.0, ay = 1.0)

    examplefunc = {'name':"Gaussian (QHO Ground State)",
                   'func': np.abs(QHO_Ground_State(X, Y))**2}

    from scipy.special import i0

    def h_analytic(q, sigma = 1):
        return np.exp(-sigma**2 * q**2 / 4) / (2 * np.pi)

    def HT_analytic(delta, sigma = 1, epsilon = 1):
        x = (delta**2) / (4 * sigma**2)
        prefactor = 1 / (epsilon * sigma * np.sqrt(8*np.pi))
        result = prefactor * np.exp(-x) * i0(x)
        return result

    a = 2.0
    deltas = np.array([[0, 0], 
                       [1, 0],
                       [2, 0],
                       [3, 0],
                       [4, 0],
                       ])
    
    Amat = np.array([[a, 0],
                     [0, a]])

    deltas = deltas @ Amat

    rmax = 10.0

    ti_cdh = time()
    f = g2d.PolarDecomposition(f_xy = examplefunc['func'], 
                              x = x, 
                              y = y, 
                              Nr = 256, 
                              Ntheta = 512,
                              rmax = rmax, 
                              recon_err_tol = 1.0, 
                              recon_power_tol = 1e-15, 
                              m_abs_max = None,
                              interp_method = 'cubic')
    tf_cdh = time()
    print(f'Decomposition time: {tf_cdh - ti_cdh}')

    N = 1024 * 10
    ti_hoh = time()
    F = q2d.HankelofHarmonics(cdh = f, 
                          q_max = 20*np.pi/f.r_cutoff[0], 
                          N = N, 
                          h = np.pi/N, 
                          N_q = 512*2, 
                          tol_power = 1e-15, 
                          # zeropadparams = (3, 1024), 
                          # eta_bump = eta,
                          tol_tot_error = 1.0,
                          interp_method = 'cubic'
                          )
    tf_hoh = time()
    print(f'Hankel transform time: {tf_hoh - ti_hoh}')
    
    print(f'Hankel transform roundtrip error: {F.total_error:.4f}%')
    plt.plot(F.q[0], np.abs(F(0, F.q[0])), label = 'Analytic', lw = 3)
    plt.plot(F.q[0], np.abs(h_analytic(F.q[0])), label = 'Numerical', lw = 3, ls = '--')
    plt.xlabel(r"$q~[\mathrm{\AA}^{-1}]$")
    plt.ylabel(r"$|F_m(q)|$")
    plt.title('Hankel transforms')
    plt.legend()
    plt.show()
    plt.close()
        
    ti_V = time()
    V = q2d.Interact(deltas, F, F, U_q, N = 1024*2)
    tf_V = time()
    print(f'Interaction calculation time: {tf_V - ti_V}')

    print()
    print('distance [Ang]      Analytic [e^2]       Numerical[e^2]      % Error')
    print('-'*60)
    for i, d in enumerate(deltas):
        d_mag = np.sqrt((d**2).sum())
        V_approx = V._V[i].real
        V_analytic = 2*np.pi*HT_analytic(d_mag)
        error = np.abs((V_analytic - V_approx)/V_analytic)*100
        print(f'{d_mag}'+' '*18, f'{V_analytic:.4e}' + " "*10 , f"{V_approx:.4e}", " "*7 + f"{error:.2e} %")
